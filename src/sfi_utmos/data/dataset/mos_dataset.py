import random
from pathlib import Path

import torch
import torchaudio
from lightning import LightningDataModule


class MOSDataModule(LightningDataModule):
    """
    Data module for Mean Opinion Score (MOS) dataset.
    This module handles the loading and preparation of MOS data for training and validation.
    """

    def __init__(
        self,
        train_mos_data_path,
        valid_mos_data_path,
        wav_root,
        batch_size=32,
        shuffle=False,
    ):
        """
        Initialize the MOSDataModule.

        Args:
            train_mos_data_path (str): Path to the training MOS data file.
            valid_mos_data_path (str): Path to the validation MOS data file.
            wav_root (str): Root directory for audio files.
            batch_size (int, optional): Batch size for data loaders. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        """
        super().__init__()
        self.train_mos_data_path = train_mos_data_path
        self.valid_mos_data_path = valid_mos_data_path
        self.wav_root = wav_root
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage):
        self.train_dataset = MOSDataset(
            self.train_mos_data_path,
            self.wav_root,
            shuffle=True,
        )
        self.valid_dataset = MOSDataset(
            self.valid_mos_data_path,
            self.wav_root,
            shuffle=False,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=16,
            collate_fn=self.collate_fn,  # No special collate function needed
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            collate_fn=self.collate_fn,  # No special collate function needed
        )

    def collate_fn(self, samples):
        waveforms = [sample[0] for sample in samples]
        sample_rates = [sample[1] for sample in samples]
        listener_ids = [sample[2] for sample in samples]
        mos_scores = [sample[3] for sample in samples]
        return (
            waveforms,
            torch.tensor(sample_rates),
            torch.tensor(listener_ids),
            torch.tensor(mos_scores),
        )


class MOSDataset(torch.utils.data.Dataset):
    """
    Dataset for Mean Opinion Score (MOS) data.
    This dataset is designed to handle audio files and their corresponding MOS scores.
    """

    def __init__(
        self,
        mos_data_path,
        wav_root,
        shuffle=False,
    ):
        """
        Initialize the MOSDataset.

        Args:
            mos_data (list): List of tuples containing (audio_path, mos_score).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(mos_data_path, "r") as f:
            mos_data = [line.strip().split(",") for line in f.readlines()]
        self.mos_data = mos_data
        if shuffle:
            random.shuffle(self.mos_data)
        self._wav_root = wav_root

    def __len__(self):
        return len(self.mos_data)

    def __getitem__(self, idx):
        audio_path, listener_id, mos_score = self.mos_data[idx]
        listener_id = int(listener_id)
        mos_score = float(mos_score)
        waveform, sample_rate = torchaudio.load(Path(self._wav_root) / audio_path)

        return waveform.view(-1), sample_rate, listener_id, mos_score
