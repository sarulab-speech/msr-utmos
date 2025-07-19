# SFI-UTMOS

**Sampling-Frequency-Independent Convolution for MOS Prediction**

---

## Overview

Most modern audio-based Self-Supervised Learning (SSL) models require a fixed input sample rate (e.g., 16kHz). This necessitates resampling all input audio, which can be computationally expensive and potentially degrade signal quality.

**SFI-UTMOS** introduces a model that is independent of the sampling frequency for the task of Mean Opinion Score (MOS) prediction. It leverages knowledge distillation to train a flexible "student" model that can process audio at various native sample rates (e.g., 16kHz, 24kHz, 48kHz). This student model is then fine-tuned for high-quality MOS prediction without the need for resampling during inference.

## Key Features

-   **Sampling-Frequency-Independent:** Processes audio files at their native sample rates.
-   **Knowledge Distillation:** A smaller, flexible student model learns from a powerful, pre-trained teacher model (e.g., Wav2Vec2-Base).
-   **High-Quality MOS Prediction:** Achieves strong performance on MOS prediction tasks by fine-tuning the distilled model.
-   **Efficient & Modern:** Built with PyTorch Lightning for reproducible and scalable training.
-   **CLI-Driven:** Uses `lightning.pytorch.cli` for easy configuration and execution of training runs from the command line.

## Project Structure

```
sfi-utmos/
├── notebooks/
│   └── predict.py           # Script for running inference
├── src/
│   └── sfi_utmos/
│       ├── data/            # Data loading and processing
│       │   └── dataset/
│       │       ├── glob_wav_dataset.py  # Dataloader for distillation
│       │       └── mos_dataset.py       # Dataloader for MOS fine-tuning
│       ├── model/
│       │   ├── distill_ssl.py # LightningModule for distillation
│       │   └── ssl_mos.py     # LightningModule for MOS prediction
│       ├── train.py             # Main script for distillation training
│       └── train_mos.py         # Main script for MOS fine-tuning
├── .gitignore
├── README.md
└── uv.lock                  # Project dependencies
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sfi-utmos/sfi-utmos.git
    cd sfi-utmos
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:** This project uses `uv` for fast dependency management. The `uv.lock` file ensures a reproducible environment.
    ```bash
    pip install uv
    uv pip install -e .[dev]
    ```

## Usage

The project is structured around two main stages: (1) Distilling the SFI model and (2) Fine-tuning for MOS prediction.

### 1. Distillation Training

First, train the sampling-frequency-independent student model using `src/sfi_utmos/train.py`. You will need a configuration file to specify models, data paths, and training parameters.

**Example Configuration (`distill_config.yaml`):**
```yaml
model:
  class_path: sfi_utmos.model.distill_ssl.DistillSSL
  init_args:
    teacher_model_path_or_name: facebook/wav2vec2-base
    student_model_path_or_name: sfi/wav2vec2-base-sfi # Example student model
    config:
      lr: 1e-4
      weight_decay: 1e-5
      sample_rates: [16000, 24000, 48000]

data:
  class_path: sfi_utmos.data.dataset.glob_wav_dataset.GlobWavDataModule
  init_args:
    data_dir: /path/to/your/unlabeled/wav/files
    batch_size: 16

trainer:
  max_epochs: 10
  accelerator: gpu
  devices: 1
```

**Run Training:**
```bash
python src/sfi_utmos/train.py fit --config distill_config.yaml
```

### 2. MOS Prediction Fine-tuning

Once you have a trained SFI model checkpoint, fine-tune it for MOS prediction using `src/sfi_utmos/train_mos.py`.

**Run Training:**
```bash
python src/sfi_utmos/train_mos.py fit --model.ssl_model_path=/path/to/distilled_model --data.train_mos_data_path=/path/to/train.csv --data.valid_mos_data_path=/path/to/val.csv --data.wav_root=/path/to/wavs --trainer.max_epochs=5
```

### 3. Inference

Use the `notebooks/predict.py` script to generate MOS predictions for a directory of audio files using a trained checkpoint.

```bash
python notebooks/predict.py \
    --ckpt_path /path/to/your/mos_model.ckpt \
    --wav_dir /path/to/inference/wavs \
    --id my_prediction_run \
    --device cuda
```

Predictions will be saved in `predictions_final/my_prediction_run/answer.txt`.
