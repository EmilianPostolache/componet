# CompoNet
## Introduction
This is the official repository of the CompoNet baseline defined in [COCOLA: Coherence-Oriented Contrastive Learning of Musical Audio Representations](https://arxiv.org/abs/2404.16969).
The code for the proper COCOLA model can be found at https://github.com/gladia-research-group/cocola.

## Installation
### Create virtual environment (Optional)

```
conda create --name componet python=3.11
conda activate componet
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Pretrained Models

| Model Checkpoint                                                                                           | Train Dataset                                                                                                                                              | Train Config                            | Description                                                                                                                              |
|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| [musdb-conditional_epoch=423.ckpt](https://drive.google.com/drive/folders/15NWq91DpYaooEqQQtrZdov9vUa5TxcXP?usp=sharing)     | [MusDB](https://sigsep.github.io/datasets/musdb.html) | `exp/train_musdb_conditional.yaml` | CompoNet model trained on MusDB dataset using AudioLDM2-large as base model, finetuing ControlNet adapter. |
| [moisesdb-conditional_epoch=250.ckpt](https://drive.google.com/drive/folders/1jBpCGQymvK3_-28m1P8IxKwNFqVORlqe?usp=sharing)  | [MoisesDB](https://github.com/moises-ai/moises-db) | `exp/train_moisesdb_conditional.yaml` | CompoNet model trained on MoisesDB dataset using AudioLDM2-large as base model, finetuing ControlNet adapter.   |
| [slakh-conditional_epoch=93.ckpt](https://drive.google.com/drive/folders/1Cpv_7elu2BvZNJW3pXQcKMqxDoMqJP6k?usp=sharing)      | [Slakh2100](http://www.slakh.com/) | `exp/train_slakh_conditional_attentions.yaml` | CompoNet model trained on Slakh2100 dataset using AudioLDM2-large as base model, finetuing ControlNet adapter and UNet cross-attentions. |

### Inference example

Inference can be performed using `inference.ipynb`. The model is first instantiated and the checkpoint loaded. Specify
the model config (the `Train Config` in the table above without `.yaml` extension) as `exp_cfg` and checkpoint path in `ckpt_path`.  

Then, load your input with:
```python
y, sr = torchaudio.load("in.wav") # load you audio input
```
And specify the inference `prompt`.

Full example with `musdb-conditional`:

```python
import hydra
import torch
import torchaudio

exp_cfg = "train_musdb_conditional"
ckpt_path = "../ckpts/musdb-conditional_epoch=423.ckpt"

with hydra.initialize(config_path="..", version_base=None):
    cond_cfg = hydra.compose(config_name="config", overrides=[f'exp={exp_cfg}'])
model = hydra.utils.instantiate(cond_cfg["model"])

ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.cuda()

y, sr = torchaudio.load("in.wav") # load you audio input
prompt = "in: other_1, vocals_1; out: vocals_1"

assert sr == 16000

y = torch.clip(y, -1, 1)
y_melspec = model.stft.cpu().mel_spectrogram(y.cpu())[0]
y_latents = model.vae.encode(y_melspec.cuda().permute(0, 2, 1).unsqueeze(1)).latent_dist.sample()
y_latents = y_latents * model.vae.config.scaling_factor

samples = model.pipeline([prompt], num_inference_steps=150,
         guidance_scale=1.0, audio_length_in_s=10.23, controlnet_cond=y_latents.cuda()).audios
torchaudio.save(f"out.wav", torch.tensor(samples[0]).unsqueeze(0), sample_rate=sr)
```

For `musdb-conditional` and `slakh-conditional` the prompts do not have a genre attribute. For example:

```python
prompt = "in: other_1, vocals_1; out: vocals_1"
```

For `moisesdb-conditional` you have to specify a lowercase genre (e.g., `pop`, `rock`) preceding the input and output
tags:

```python
prompt = "genre: pop; in: guitar_1, vocals_1; out: other_keys_1, drums_1"
```

### Tags

The available stem tags for `musdb-conditional` are

```python
STEMS = ['bass', 'drums', 'vocals', 'other']
```

The available stem tags for `slakh-conditional` are

```python
STEMS = ['bass', 'drums', 'guitar', 'piano']
```

The available stem and genre tags for `moisesdb-conditional` are

```python
STEMS = ['bass', 'bowed_strings', 'drums', 'guitar', 'other', 'other_keys', 'other_plucked', 'percussion', 'piano', 'vocals', 'wind']
GENRES = ['blues', 'bossa_nova', 'country', 'electronic', 'jazz', 'musical_theatre', 'pop', 'rap', 'reggae', 'rock', 'singer_songwriter', 'world_folk']
```

### Training example

TODO