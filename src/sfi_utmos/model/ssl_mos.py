from pathlib import Path
from typing import Literal, Optional

import schedulefree
import torch
import torchaudio
import transformers
from lightning import LightningModule


class SSLMOSLightningModule(LightningModule):
    """
    Lightning module for training and evaluating SSL MOS models.
    This module handles the training, validation, and testing of the model.
    """

    def __init__(
        self,
        ssl_model_path,
        pretrained_model_path: Optional[str] = None,
        processor_path: Optional[str] = None,
        condition_sr: Optional[bool] = False,
        ssl_model_type: Optional[Literal["w2v2", "hubert", "wavlm"]] = None,
    ):
        """
        Initialize the SSLMOSLightningModule.

        Args:
            model: The model to be trained.
            criterion: Loss function to be used.
            optimizer: Optimizer for training.
            scheduler: Learning rate scheduler (optional).
        """
        super().__init__()
        if not Path(ssl_model_path).exists():
            if ssl_model_type is None:
                raise ValueError(
                    "ssl_model_type must be specified if ssl_model_path does not exist."
                )
            if ssl_model_type == "w2v2":
                ssl_model_path = "Wataru/sfi_w2v2_distill_ears"
            elif ssl_model_type == "hubert":
                ssl_model_path = "Wataru/sfi_hubert_distill_ears"
            elif ssl_model_type == "wavlm":
                ssl_model_path = "Wataru/sfi_wavlm_distill_ears"
        self.ssl_model = transformers.AutoModel.from_pretrained(
            ssl_model_path, trust_remote_code=True
        ).train()
        _processor_path = (
            processor_path if processor_path is not None else ssl_model_path
        )
        self.processor = transformers.AutoProcessor.from_pretrained(
            _processor_path, trust_remote_code=True
        )
        self.pred_liner = torch.nn.Sequential(
            torch.nn.Linear(
                self.ssl_model.config.hidden_size, self.ssl_model.config.hidden_size
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(self.ssl_model.config.hidden_size, 1),
        )

        self.listenr_embedding = torch.nn.Embedding(
            1000, self.ssl_model.config.hidden_size
        )
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()
        self._pretrained_model_path = pretrained_model_path
        if pretrained_model_path is not None:
            weight = torch.load(pretrained_model_path, map_location=self.device)
            self.load_state_dict(weight["state_dict"])
        self.is_sfi = hasattr(self.ssl_model, "set_sample_rate")
        if condition_sr:
            self.sr_embedding = torch.nn.Embedding(3, self.ssl_model.config.hidden_size)
            self.sr2id = {16000: 0, 24000: 1, 48000: 2}
        self.condition_sr = condition_sr

    def forward(self, waves, listenr_ids: torch.Tensor, srs=None):
        """
        Forward pass through the model.

        Args:
            x: Input data.

        Returns:
            Output of the model.
        """
        if listenr_ids.ndim != 1:
            raise ValueError("shape error")
        if not self.is_sfi:
            waves = self.processor(
                [w.detach().cpu().numpy() for w in waves],
                return_tensors="pt",
                sampling_rate=16000,
                padding=True,
            ).to(self.device)
            outputs = self.ssl_model(**waves)
        else:
            outputs = self.ssl_model(
                input_values=waves,
            )
        hidden_states = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, seq_len, hidden_size)
        if self.condition_sr:
            hidden_states = hidden_states + self.sr_embedding(srs).unsqueeze(1)
        hidden_states = hidden_states + self.listenr_embedding(listenr_ids).unsqueeze(1)
        logits = torch.tanh(
            self.pred_liner(hidden_states).mean(dim=1)
        )  # Shape: (batch_size, 1)
        return logits.squeeze(-1)

    def step(self, batch, batch_idx, stage="train"):
        """
        Training/Validation step for the model.

        Args:
            batch: A batch of data.
            batch_idx: Index of the batch.

        Returns:
            loss: Computed loss for the batch.
        """
        waves, srs, listener_ids, mos = batch
        mos = (mos - 3) / 2  # Normalize MOS to [-1, 1]
        if srs.unique().numel() > 1:
            raise ValueError("All samples in a batch must have the same sample rate.")
        if self.is_sfi:
            self.ssl_model.set_sample_rate(srs[0].item())
            waves = torch.nn.utils.rnn.pad_sequence(
                [w.view(-1) for w in waves], batch_first=True
            ).to(self.device)
        else:
            waves = [
                torchaudio.functional.resample(w.view(-1), srs[0].item(), 16000)
                for w in waves
            ]
        if self.condition_sr:
            srs = torch.stack(
                [torch.tensor(self.sr2id[sr.detach().cpu().item()]) for sr in srs]
            ).to(self.device)
        logits = self.forward(waves, listener_ids, srs)
        loss = self.criterion(logits, mos)
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            batch_size=srs.size(0),
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            optimizer: Configured optimizer.
        """
        optimizer = schedulefree.RAdamScheduleFree(self.parameters(), lr=1e-4)
        return optimizer

    def set_optimizer_state(self, state: str):
        opts = self.optimizers()
        if not isinstance(opts, list):
            opts = [opts]

        for opt in opts:
            if isinstance(opt, schedulefree.RAdamScheduleFree):
                if state == "train":
                    opt.train()
                elif state == "eval":
                    opt.eval()
                else:
                    raise ValueError(f"Unknown train state {state}")

    def on_train_epoch_start(self) -> None:
        self.set_optimizer_state("train")

    def on_validation_epoch_start(self) -> None:
        self.set_optimizer_state("eval")
