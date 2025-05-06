import pathlib

import torch
import torchaudio
import transformers
from lightning import LightningDataModule


class GlobWavDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, preprocessor=None, max_wav_secs=20):
        self.data_dir = data_dir
        self.transform = transform
        self.wav_files = [f for f in pathlib.Path(data_dir).glob("**/*.wav")]
        if preprocessor is not None:
            self.preprocessor = transformers.AutoProcessor.from_pretrained(preprocessor)
        else:
            self.preprocessor = None
        self.max_wav_secs = max_wav_secs

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        # truncate waveforms longer than max_wav_secs at random position
        if waveform.size(1) > self.max_wav_secs * sample_rate:
            start = torch.randint(
                0, waveform.size(1) - int(self.max_wav_secs * sample_rate), (1,)
            ).item()
            end = start + int(self.max_wav_secs * sample_rate)
            waveform = waveform[:, start:end]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.view(-1), sample_rate


class GlobWavDataModule(LightningDataModule):
    """
    Data module for the Glow Wav dataset.
    """

    def __init__(self, data_dir, batch_size=32, num_workers=4, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        dataset = GlobWavDataset(self.data_dir, transform=self.transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, (0.9, 0.1)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-length sequences.
        """
        # Assuming all tensors in the batch are of the same shape
        # assume that the samplign rate is the same for all files
        assert all(x[1] == batch[0][1] for x in batch), (
            "All samples must have the same sampling rate"
        )

        return torch.nn.utils.rnn.pad_sequence(
            [x[0] for x in batch], batch_first=True
        ), batch[0][1]
