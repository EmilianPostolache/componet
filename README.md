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

| Model Checkpoint                                                                                           | Train Dataset                                                                                                                                              | Train Config                              | Description                                                                      |
|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------|
| [componet_musdb](https://drive.google.com/drive/folders/15NWq91DpYaooEqQQtrZdov9vUa5TxcXP?usp=sharing)     | [MusDB](https://sigsep.github.io/datasets/musdb.html)                                                                                              | `exp/train_musdb_conditional.yaml`        | CompoNet model trained on MusDB dataset using AudioLDM2-large as base model.     |
| [componet_moisesdb](https://drive.google.com/drive/folders/1jBpCGQymvK3_-28m1P8IxKwNFqVORlqe?usp=sharing)  | [MoisesDB](https://github.com/moises-ai/moises-db) | `configs/train_moisesdb_conditional.yaml` | CompoNet model trained on MoisesDB dataset using AudioLDM2-large as base model.  |

Slakh2100 model checkpoint coming soon...

### Inference example

Inference can be performed using `inference.ipynb`. After loading your input with:

```jupyter
y, sr = torchaudio.load("in.wav") # load you audio input
```

Run the `model.pipeline` line:

- Example for `componet_musdb` (do not use a genre in the prompt):

    ```juputer
    samples = model.pipeline(["in: vocals_1, other_1; out: vocals_1"], num_inference_steps=150,
             guidance_scale=1.0, audio_length_in_s=10.23, controlnet_cond=y_latents.cuda()).audios
    ```

- Example for `componet_musdb` (use genre in the prompt):

    ```juputer
    samples = model.pipeline(["pop; in: vocals_1, guitar_1; out: drums_1"], num_inference_steps=150,
             guidance_scale=1.0, audio_length_in_s=10.23, controlnet_cond=y_latents.cuda()).audios
    ```

### Training example

TODO