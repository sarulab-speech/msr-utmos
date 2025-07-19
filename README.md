# MSR-UTMOS

**Multi-Sampling-Frequency Naturalness MOS Prediction Using Self-Supervised Learning Model with Sampling-Frequency-Independent Layer**

Authors
-  Go Nishikawa
-  [Wataru Nakata](https://wataru-nakata.github.io/)
-  [Yuki Saito](https://sython.org/)
-  [Kanami Imamura](https://scholar.google.com/citations?user=Lw11ESIAAAAJ&hl=en)
-  [Hiroshi Saruwatari](https://scholar.google.com/citations?hl=en&user=OS1XAoMAAAAJ)
-  [Tomohiko Nakamura](https://tomohikonakamura.github.io/Tomohiko-Nakamura/index.html)

---

## Abstract 

We introduce our submission to the AudioMOS Challenge (AMC) 2025 Track 3: mean opinion score (MOS) prediction for speech with multiple sampling frequencies (SFs).
Our submitted model integrates an SF-independent (SFI) convolutional layer into a self-supervised learning (SSL) model to achieve SFI speech feature extraction for MOS prediction.
We present some strategies to improve the MOS prediction performance of our model: distilling knowledge from a pretrained non-SFI-SSL model and pretraining with a large-scale MOS dataset.
Our submission to the AMC 2025 Track 3 ranked the first in one evaluation metric and the fourth in the final ranking.
We also report the results of our ablation study to investigate essential factors of our model.


## Project Structure

```
sfi-utmos/
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
    uv sync
    ```

## Usage

The project is structured around two main stages: (1) Distilling the SFI model and (2) Fine-tuning for MOS prediction.

### 1. Distillation Training

First, train the sampling-frequency-independent student model using `src/sfi_utmos/train.py`. You will need a configuration file to specify models, data paths, and training parameters.

**Example Configuration (`distill_config.yaml`):**
```yaml
model:
  class_path: sfi_utmos.model.distill_ssl.DistillSSL # The module is sfi-utmos, but the model is msr-utmos
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

