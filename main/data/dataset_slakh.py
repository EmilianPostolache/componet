from functools import partial
from typing import List, Optional
import yaml

import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset.autodecode import torch_audio
from torchaudio.functional import resample
import random


def _weights_for_nonzero_refs(source_waveforms):
  """Return shape (batch, source) weights for signals that are nonzero."""
  source_norms = torch.sqrt(torch.mean(source_waveforms ** 2, dim=-1))
  return torch.greater(source_norms, 1e-8)

def _fn_resample(sample, sample_rate):
    sample_rate_orig = sample[-1]
    return ({stem: resample(track, orig_freq=sample_rate_orig, new_freq=sample_rate)
            for stem, track in sample[0].items()}, sample_rate)

def yaml_bytes_to_dict(input_bytes):
    # Decode bytes to string
    string = input_bytes.decode("utf-8")
    # Parse string as YAML
    return yaml.safe_load(string)

def decode_yaml(sample):
    # Convert metadata.yaml entry to dict
    sample["metadata.yaml"] = yaml_bytes_to_dict(sample["metadata.yaml"])
    return sample

def _fill_missing_keys_and_pad(sample):
    stems = ['bass', 'drums', 'guitar', 'piano']
    # Extract stem ids and stem names, consider only stems of interest
    stems_in_sample = [(stem_id, attributes['inst_class'].lower())
                       for stem_id, attributes in sample['metadata.yaml']['stems'].items()
                       if attributes['inst_class'].lower() in stems and attributes['midi_saved']
                       and attributes['audio_rendered']]

    if not stems_in_sample:
        raise ValueError("No stems found in sample")

    # Prepare stem data
    try:
        stem_to_data = {stem: sample[f"{stem_id.lower()}.flac"] for stem_id, stem in stems_in_sample}
    except KeyError as error:
        print(sample.items())
        print(stems_in_sample)
        raise error

    # Default tensor and sample rate for missing stems
    _, (_, default_sr) = next(iter(stem_to_data.items()))
    # stem_to_data = {stem: stem_to_data.get(stem, (torch.zeros_like(default_tensor), default_sr)) for stem in stems}

    # Compute max length
    max_length = max(tensor.size(1) for tensor, _ in stem_to_data.values())

    # Pad data
    stem_to_data = ({stem + "_1": F.pad(tensor, (0, max_length - tensor.size(1))) for stem, (tensor, sr) in
                    stem_to_data.items()}, default_sr)

    return stem_to_data


def _get_slices(src, chunk_dur):
    for sample in src:
        # get length of first element in step
        stems, sr = sample
        channels, length = list(stems.values())[0].shape
        chunk_size = int(sr * chunk_dur)

        # Pad signals to chunk_size if they are shorter
        if length < chunk_size:
             padding = torch.zeros(channels, chunk_size - length)
             stems = {stem: (torch.cat([track, padding], dim=-1), stem_type)
                      for stem, (track, stem_type) in stems}
             length = chunk_size

        max_shift = length - (length // chunk_size) * chunk_size
        shift = torch.randint(0, max_shift + 1, (1,)).item()

        for i in range(length // chunk_size):
            start_idx = min(length - chunk_size, i * chunk_size + shift)
            end_idx = start_idx + chunk_size

            chunks = {stem: track[:, start_idx: end_idx] for stem, track in stems.items()}

            if channels == 2:
                raise ValueError("Only one channel is supported")

            chunks = {k: v for k, v in chunks.items() if _weights_for_nonzero_refs(v)}
            if len(chunks) == 0:
                continue

            yield chunks

def create_slakh_dataset(
        path: str,
        sample_rate: int,
        chunk_dur: Optional[float] = None,
        shardshuffle: bool = False):

    fill_missing_keys_and_pad = partial(_fill_missing_keys_and_pad)
    get_slices = partial(_get_slices, chunk_dur=chunk_dur)
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)

    # create datapipeline
    dataset = (wds.WebDataset(path, shardshuffle=shardshuffle).decode(torch_audio).map(decode_yaml).
                              map(fill_missing_keys_and_pad).map(fn_resample))
    dataset = dataset.compose(get_slices) if chunk_dur is not None else dataset
    return dataset


def collate_fn_conditional(samples):
    subsets = [(random.sample(list(range(len(sample))), k=random.randint(1, len(sample))),
               random.sample(list(range(len(sample))), k=random.randint(1, len(sample)))) for sample in samples]

    xs = []
    ys = []
    zs = []

    default_shape = list(samples[0].values())[0].shape

    for subset_pair, sample in zip(subsets, samples):
        stems = sample
        stem_keys = list(stems.keys())
        in_indices, out_indices = subset_pair
        in_stems_prompt = [stem_keys[i] for i in in_indices]
        out_stems_prompt = [stem_keys[i] for i in out_indices]
        in_track = torch.cat([stems[stem_keys[i]] for i in in_indices], dim=0).sum(dim=0, keepdim=True) #\
                           #  if in_indices else torch.zeros(default_shape)
        out_track = torch.cat([stems[stem_keys[i]] for i in out_indices], dim=0).sum(dim=0, keepdim=True)
        xs.append(out_track)
        ys.append(in_track)
        zs.append(f"in: {', '.join(in_stems_prompt)}; out: {', '.join(out_stems_prompt)}")

    return torch.concat(xs), torch.concat(ys), zs


if __name__ == '__main__':
    dataset_train = create_slakh_dataset("../../train-0.tar",
                                                  sample_rate=16000, chunk_dur=10.23, shardshuffle=True)
    import sounddevice as sd
    dataloader = DataLoader(dataset_train,
                            batch_size=16,
                            pin_memory=False,
                            drop_last=True,
                            collate_fn=collate_fn_conditional,
                            num_workers=0)
    data = next(iter(dataloader))