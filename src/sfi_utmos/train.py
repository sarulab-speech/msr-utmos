from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from sfi_utmos.data.dataset.glob_wav_dataset import GlobWavDataModule  # noqa: F407
from sfi_utmos.model.distill_ssl import DistillSSL  # noqa: F407


def cli_main():
    cli = LightningCLI(DistillSSL, GlobWavDataModule)


if __name__ == "__main__":
    cli_main()
