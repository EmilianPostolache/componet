import random
from typing import List, Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

from main.dependencies.audioldm2 import utils as audioldm2_utils
from main.dependencies.audioldm2.utilities.audio.stft import TacotronSTFT
from main.dependencies.diffusers.pipelines.audioldm2.modeling_audioldm2 import (AudioLDM2UNet2DConditionModel,
                                                                                CrossAttnDownBlock2D,
                                                                                CrossAttnUpBlock2D,
                                                                                UNetMidBlock2DCrossAttn)
from diffusers import (AudioLDM2ProjectionModel, AutoencoderKL, DDIMScheduler)
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, ClapModel, T5EncoderModel, GPT2Model, SpeechT5HifiGan, RobertaTokenizer,
                          RobertaTokenizerFast, AutoFeatureExtractor)

from main.conditional.controlnet import ControlNetAudioLDM2Model
from main.conditional.pipeline import ControlNetAudioLDM2Pipeline
from main.utils import log_wandb_audio_batch, log_wandb_audio_spectrogram


def prepare_inputs_for_generation(inputs_embeds,
                                  attention_mask=None,
                                  past_key_values=None,
                                  **kwargs,
                                  ):
    if past_key_values is not None:
        # only last token for inputs_embeds if past is defined in kwargs
        inputs_embeds = inputs_embeds[:, -1:]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }


""" Model """

class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        pretrained_model_name_or_path: str,
        finetune_unet_attentions: bool = False,
        train_unet_scratch: bool = False,
        p_uncond=0.2
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.finetune_unet_attentions = finetune_unet_attentions
        self.train_unet_scratch = train_unet_scratch

        self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
        self.tokenizer_2 = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
        self.text_encoder = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.text_encoder_2 = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
        self.projection_model = AudioLDM2ProjectionModel.from_pretrained(pretrained_model_name_or_path, subfolder="projection_model")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        self.vocoder = SpeechT5HifiGan.from_pretrained(pretrained_model_name_or_path, subfolder="vocoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        if train_unet_scratch:
            self.unet = AudioLDM2UNet2DConditionModel.from_config(pretrained_model_name_or_path, subfolder="unet")
        else:
            self.unet = AudioLDM2UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        self.controlnet = ControlNetAudioLDM2Model.from_unet(self.unet)

        self.pipeline = ControlNetAudioLDM2Pipeline(self.vae, self.text_encoder, self.text_encoder_2,
                                                    self.projection_model, self.feature_extractor, self.tokenizer,
                                                    self.tokenizer_2, self.unet, self.noise_scheduler, self.vocoder,
                                                    self.controlnet)

        audio_config = audioldm2_utils.get_basic_config()
        self.stft = TacotronSTFT(
            audio_config["preprocessing"]["stft"]["filter_length"],
            audio_config["preprocessing"]["stft"]["hop_length"],
            audio_config["preprocessing"]["stft"]["win_length"],
            audio_config["preprocessing"]["mel"]["n_mel_channels"],
            audio_config["preprocessing"]["audio"]["sampling_rate"],
            audio_config["preprocessing"]["mel"]["mel_fmin"],
            audio_config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.vae.requires_grad_(False)
        if not self.finetune_unet_attentions and not self.train_unet_scratch:
            self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vocoder.requires_grad_(False)
        self.projection_model.requires_grad_(False)
        # self.language_model.requires_grad_(False)
        self.stft.requires_grad_(False)
        self.p_uncond = p_uncond

    def configure_optimizers(self):
        params = list(self.controlnet.parameters())
        if self.finetune_unet_attentions:
            blocks = list(self.unet.down_blocks) + [self.unet.mid_block] + list(self.unet.up_blocks)
            for b in blocks:
                if isinstance(b, (CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn, CrossAttnUpBlock2D)):
                    params += list(b.attentions.parameters())
                    b.resnets.requires_grad_(False)
                else:
                    b.requires_grad_(False)
        elif self.train_unet_scratch:
            params += list(self.unet.parameters())
        optimizer = torch.optim.AdamW(
            params,
            # list(self.projection_model.parameters()) +
            # list(self.language_model.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def step(self, batch):
        x, y, z, w = batch

        x = torch.clip(x, -1,1)
        x_melspec = self.stft.cpu().mel_spectrogram(x.cpu())[0]
        x_latents = self.vae.encode(x_melspec.cuda().permute(0, 2, 1).unsqueeze(1)).latent_dist.sample()
        x_latents = x_latents * self.vae.config.scaling_factor

        y = torch.clip(y, -1,1)
        y_melspec = self.stft.cpu().mel_spectrogram(y.cpu())[0]
        y_latents = self.vae.encode(y_melspec.cuda().permute(0, 2, 1).unsqueeze(1)).latent_dist.sample()
        y_latents = y_latents * self.vae.config.scaling_factor

        noise = torch.randn_like(x_latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                  (x_latents.shape[0],), device=x_latents.device)
        timesteps = timesteps.long()
        x_noisy_latents = self.noise_scheduler.add_noise(x_latents, noise, timesteps)
        tag_embed, tag_attention_mask = self.encode_tags(prompt=z)
        if random.random() < self.p_uncond:
            clap_embed, clap_attention_mask = self.encode_claps(w=[torch.zeros(1, 163680)] * x.shape[0],
                                                                   tag_embed=tag_embed,
                                                                   tag_attention_mask=tag_attention_mask)
        else:
            clap_embed, clap_attention_mask = self.encode_claps(w=w, tag_embed=tag_embed, tag_attention_mask=tag_attention_mask)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x_noisy_latents,
            timesteps,
            controlnet_cond=y_latents,
            encoder_hidden_states=clap_embed.detach(),
            encoder_hidden_states_1=tag_embed.detach(),
            encoder_attention_mask_1=tag_attention_mask.detach(),
            return_dict=False,
        )

        model_pred = self.unet(
            x_noisy_latents,
            timesteps,
            encoder_hidden_states=clap_embed.detach(),
            encoder_hidden_states_1=tag_embed.detach(),
            encoder_attention_mask_1=tag_attention_mask.detach(),
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x_latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("valid_loss", loss)
        return loss


    def encode_tags(self, prompt):
        tokenizer = self.tokenizer_2
        text_encoder = self.text_encoder_2

        text_inputs = tokenizer(
            prompt,
            padding="max_length" if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast)) else True,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1: -1])
            print(
                f"The following part of your input was truncated because {text_encoder.config.model_type} can "
                f"only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
            )

        text_input_ids = text_input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=self.device)
        attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=self.device)
        return prompt_embeds, attention_mask

    def encode_claps(self, w, tag_embed, tag_attention_mask):
        ### encode claps
        audio_inputs = torch.stack(w, dim=0)
        audio_inputs = torchaudio.functional.resample(audio_inputs, orig_freq=16000, new_freq=48000)
        clap_embeds = []
        for i in range(audio_inputs.shape[0]):
            clap_inputs = self.feature_extractor(audio_inputs[i, 0].cpu(), return_tensors="pt", sampling_rate=48000)
            clap_embed = self.text_encoder.get_audio_features(input_features=clap_inputs['input_features'].cuda(),
                                                            is_longer=clap_inputs['is_longer'])
            clap_embeds.append(clap_embed)
        clap_embeds = torch.stack(clap_embeds, dim=0)
        clap_attention_mask = torch.ones((audio_inputs.shape[0], 1), dtype=torch.long).cuda()

        projection_output = self.projection_model(
            hidden_states=clap_embeds,
            hidden_states_1=tag_embed,
            attention_mask=clap_attention_mask,
            attention_mask_1=tag_attention_mask,
        )
        projected_prompt_embeds = projection_output.hidden_states
        projected_attention_mask = projection_output.attention_mask

        return projected_prompt_embeds, projected_attention_mask



""" Datamodule """

class WebDatasetDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size_train: int,
        batch_size_val: int,
        num_workers: int,
        pin_memory: bool,
        shuffle_size: int,
        collate_fn = None,
        drop_last: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn"

    ) -> None:
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_size = shuffle_size
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context

        train_dataset = train_dataset.shuffle(self.shuffle_size) #, initial=self.shuffle_size)
        # train_dataset = train_dataset.batched(self.batch_size, collation_fn=torch.stack)

        # This should help avoiding memory explosion with num_workers>0
        # self.shared_data = mp.Manager().Namespace()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # self.shared_data.train_dataset = train_dataset
        # self.shared_data.val_dataset = val_dataset

        self.collate_fn = collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            multiprocessing_context=self.multiprocessing_context
        )


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    print("WandbLogger not found.")
    return None


class SampleLogger(Callback):
    def __init__(
        self,
        sample_rate: int,
        chunk_dur: float,
        sampling_steps: List[int],
        embedding_scale: float,
        num_samples: int = 1
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_dur = chunk_dur
        self.sampling_steps = sampling_steps
        self.embedding_scale = embedding_scale
        self.num_samples = num_samples
        self.log_next = False

    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_next:
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        wandb_logger = get_wandb_logger(trainer).experiment

        _, y, z, w = batch
        y = torch.clip(y, -1, 1)
        y_melspec = pl_module.stft.cpu().mel_spectrogram(y.cpu())[0]
        y_latents = pl_module.vae.encode(y_melspec.cuda().permute(0, 2, 1).unsqueeze(1)).latent_dist.sample()
        y_latents = y_latents * pl_module.vae.config.scaling_factor

        for i in range(self.num_samples):
            log_wandb_audio_batch(
                logger=wandb_logger,
                id=f"true_{i}",
                samples=y[i:i+1].unsqueeze(1),
                sampling_rate=self.sample_rate,
                caption=f"Prompt: {z[i]}",
            )
            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id=f"true_{i}",
                samples=y[i:i+1].unsqueeze(1),
                sampling_rate=self.sample_rate,
                caption=f"Prompt: {z[i]}",
            )

        for steps in self.sampling_steps:
            samples = pl_module.pipeline(prompt=z,
                                         w=w,
                                         controlnet_cond=y_latents,
                                         num_waveforms_per_prompt=1,
                                         audio_length_in_s=self.chunk_dur,
                                         num_inference_steps=steps,
                                         guidance_scale=self.embedding_scale).audios
            for i in range(samples.shape[0]):
                log_wandb_audio_batch(
                    logger=wandb_logger,
                    id=f"sample_{i}",
                    samples=torch.tensor(samples[i:i+1]).unsqueeze(1),
                    sampling_rate=self.sample_rate,
                    caption=f"Sampled in {steps} steps. Prompt: {z[i]}",
                )
                log_wandb_audio_spectrogram(
                    logger=wandb_logger,
                    id=f"sample_{i}",
                    samples=torch.tensor(samples[i:i+1]).unsqueeze(1),
                    sampling_rate=self.sample_rate,
                    caption=f"Sampled in {steps} steps. Prompt: {z[i]}",
                )

        if is_train:
            pl_module.train()
