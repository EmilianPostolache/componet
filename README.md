# CompoNet
## Introduction
This is the official repository of the CompoNet baseline defined in [COCOLA: Coherence-Oriented Contrastive Learning of Musical Audio Representations](https://arxiv.org/abs/2404.16969).

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

### Install datasets
If you wish to use MoisesDB for training/validation/test, download it from the [official website](https://music.ai/research/) and unzip it inside `data`.
The other datasets ([CocoChorales](https://magenta.tensorflow.org/datasets/cocochorales), [Slakh2100](http://www.slakh.com), [Musdb](https://sigsep.github.io/datasets/musdb.html)) are automatically downoladed and extracted by the respective PyTorch Datasets.
