import random
from collections import defaultdict
from functools import partial
from typing import List, Optional

import torch
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset.autodecode import torch_audio
from torchaudio.functional import resample
import torch.nn.functional as F


def _fn_resample(sample, sample_rate):
    sample_rate_orig = sample[-1]
    return ({stem: (resample(track, orig_freq=sample_rate_orig, new_freq=sample_rate), stem_type)
            for stem, (track, stem_type) in sample[0].items()}, sample[1], sample_rate)

def _weights_for_nonzero_refs(source_waveforms):
    """Return shape (batch, source) weights for signals that are nonzero."""
    source_norms = torch.sqrt(torch.mean(source_waveforms ** 2, dim=-1))
    return torch.greater(source_norms, 1e-8)


def _fn_extract_data_and_pad(sample, mix_drums=True):
    max_len = max([v[0].shape[-1] for k, v in sample.items() if k.endswith(".mp3")])
    data_dict = sample['data.json']
    genre = data_dict['genre']
    key_counts = defaultdict(int)
    unique_dict = {}
    if mix_drums:
        acc_drums = ([], [])
    for stem in data_dict['stems']:
        stem_name = stem['stemName']
        track = sample[stem['tracks'][0]['id'] + '.mp3']
        stem_type = stem['tracks'][0]['trackType']
        if stem_name == 'drums' and mix_drums:
            acc_drums[0].append(F.pad(track[0], (0, max_len - track[0].shape[-1])))
            acc_drums[1].append(stem_type)
        else:
            key_counts[stem_name] += 1
            unique_key = f"{stem_name}_{key_counts[stem_name]}"
            unique_dict[unique_key] = (F.pad(track[0], (0, max_len - track[0].shape[-1])), stem_type)
    if mix_drums and acc_drums[0]:
        unique_dict["drums_1"] = (torch.cat(acc_drums[0], dim=0).sum(dim=0, keepdim=True), ", ".join(acc_drums[1]))
    return unique_dict, genre, track[1]


def _get_slices(src, chunk_dur):
    for sample in src:
        # get length of first element in step
        stems, genre, sr = sample
        channels, length = list(stems.values())[0][0].shape
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

            chunks = {stem: (track[:, start_idx: end_idx], stem_type)
                      for stem, (track, stem_type) in stems.items() if _weights_for_nonzero_refs(track)}
            if chunks == {}:
                continue
            if channels == 2:
                raise ValueError("Only one channel is supported")

            yield chunks, genre


def create_moisesdb_dataset(
        path: str,
        sample_rate: int,
        chunk_dur: Optional[float] = None,
        shardshuffle: bool = False):

    fn_extract_data_and_pad = partial(_fn_extract_data_and_pad)
    fn_resample = partial(_fn_resample, sample_rate=sample_rate)
    get_slices = partial(_get_slices, chunk_dur=chunk_dur)

    # create datapipeline
    dataset = (wds.WebDataset(path, shardshuffle=shardshuffle).decode(torch_audio).
               map(fn_extract_data_and_pad).map(fn_resample))
    dataset = dataset.compose(get_slices) if chunk_dur is not None else dataset
    return dataset


def collate_fn_conditional(samples):
    subsets = [(random.sample(list(range(len(sample[0]))), k=random.randint(1, len(sample[0]))),
               random.sample(list(range(len(sample[0]))), k=random.randint(1, len(sample[0])))) for sample in samples]

    xs = []
    ys = []
    zs = []

    for subset_pair, sample in zip(subsets, samples):
        stems, genre = sample
        stem_keys = list(stems.keys())
        in_indices, out_indices = subset_pair
        in_stems_prompt = [stem_keys[i] for i in in_indices]
        out_stems_prompt = [stem_keys[i] for i in out_indices]
        in_track = torch.cat([stems[stem_keys[i]][0] for i in in_indices], dim=0).sum(dim=0, keepdim=True)
        out_track = torch.cat([stems[stem_keys[i]][0] for i in out_indices], dim=0).sum(dim=0, keepdim=True)
        xs.append(out_track)
        ys.append(in_track)
        zs.append(f"genre: {genre}; in: {', '.join(in_stems_prompt)}; out: {', '.join(out_stems_prompt)}")

    return torch.concat(xs), torch.concat(ys), zs


if __name__ == '__main__':
    dataset = create_moisesdb_dataset("../../data/moisesdb/{0..3}.tar",
                                    sample_rate=16000,
                                    chunk_dur=10.0)
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            pin_memory=False,
                            drop_last=True,
                            collate_fn=collate_fn_conditional,
                            num_workers=0)
    data = next(iter(dataloader))
    print(data)
