import random

import torch
import torchaudio
import transformers
import transformers.modeling_outputs
from lightning import LightningModule


class DistillSSL(LightningModule):
    def __init__(
        self,
        teacher_model_path_or_name: str,
        student_model_path_or_name: str,
        config: dict,
    ):
        """
        Initialize the DistillSSL class.

        Args:
            teacher_model_path_or_name (str): Path or name of the teacher model.
            student_model_path_or_name (str): Path or name of the student model.
            config (dict): Configuration dictionary.
        """
        super().__init__()
        self.teacher_model_path_or_name = teacher_model_path_or_name
        self.student_model_path_or_name = student_model_path_or_name
        self.config = config
        self.save_hyperparameters()
        self.student_model: transformers.Wav2Vec2Model = (
            transformers.AutoModel.from_pretrained(
                self.student_model_path_or_name, trust_remote_code=True
            )
        )
        self.teacher_model: transformers.Wav2Vec2Model = (
            transformers.AutoModel.from_pretrained(
                self.teacher_model_path_or_name, trust_remote_code=True
            )
        )
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        self.student_model.train()
        self.criterion = torch.nn.MSELoss()

    def compute_loss(
        self,
        student_outputs: transformers.modeling_outputs.BaseModelOutput,
        teacher_outputs: transformers.modeling_outputs.BaseModelOutput,
    ) -> torch.Tensor:
        """
        Compute the loss between student and teacher outputs.

        Args:
            student_outputs: Outputs from the student model.
            teacher_outputs: Outputs from the teacher model.

        Returns:
            loss: Computed loss.
        """
        pred = torch.stack(student_outputs.hidden_states)
        target = torch.stack(teacher_outputs.hidden_states)
        min_len = min(pred.size(2), target.size(2))
        loss = self.criterion(pred[:, :, :min_len], target[:, :, :min_len])
        return loss

    def step(self, batch, batch_idx, stage="train") -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.

        Returns:
            loss: The computed loss.
        """
        batched_wav, sr = batch
        with torch.inference_mode():
            new_sr = random.choice(self.config["sample_rates"])
            resampled = torchaudio.functional.resample(
                batched_wav,
                orig_freq=sr,
                new_freq=new_sr,
            )
            self.student_model.set_sample_rate(new_sr)
            batched_wav_16k = torchaudio.functional.resample(
                batched_wav,
                orig_freq=sr,
                new_freq=16000,
            )

        student_outputs = self.student_model.forward(
            input_values=resampled.clone(), output_hidden_states=True
        )
        teacher_outputs = self.teacher_model.forward(
            input_values=batched_wav_16k.clone(), output_hidden_states=True
        )
        loss = self.compute_loss(student_outputs, teacher_outputs)
        self.log(
            f"{stage}/loss",
            loss,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        return torch.optim.RAdam(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            decoupled_weight_decay=True,
        )
