from typing import Optional

import schedulefree
import torch
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
        pretrained_model_path=Optional[str],
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
        self.ssl_model = transformers.AutoModel.from_pretrained(
            ssl_model_path, trust_remote_code=True
        ).train()
        self.processor = transformers.AutoProcessor.from_pretrained(
            ssl_model_path, trust_remote_code=True
        )
        self.pred_liner = torch.nn.Linear(self.ssl_model.config.hidden_size, 1)
        self.listenr_embedding = torch.nn.Embedding(
            1000, self.ssl_model.config.hidden_size
        )
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()
        self._pretrained_model_path = pretrained_model_path
        if pretrained_model_path is not None:
            weight = torch.load(pretrained_model_path, map_location=self.device)
            self.load_state_dict(weight["state_dict"])

    def forward(self, waves, listenr_ids):
        """
        Forward pass through the model.

        Args:
            x: Input data.

        Returns:
            Output of the model.
        """
        waves = self.processor(
            [w.detach().cpu().numpy() for w in waves],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        ).to(self.device)
        outputs = self.ssl_model(**waves)
        hidden_states = (
            outputs.last_hidden_state
        )  # Shape: (batch_size, seq_len, hidden_size)
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
        if hasattr(self.ssl_model, "set_sample_rate"):
            self.ssl_model.set_sample_rate(srs[0].item())
        logits = self.forward(waves, listener_ids)
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
        self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, stage="val")

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
