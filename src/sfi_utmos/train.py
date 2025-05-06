from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from sfi_utmos.data.dataset.glob_wav_dataset import GlobWavDataModule
from sfi_utmos.model.distill_ssl import DistillSSL


def cli_main():
    cli = LightningCLI(DistillSSL, GlobWavDataModule)


if __name__ == "__main__":
    cli_main()
