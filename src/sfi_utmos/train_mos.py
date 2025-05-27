from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from sfi_utmos.data.dataset.mos_dataset import MOSDataModule
from sfi_utmos.model.ssl_mos import SSLMOSLightningModule


def cli_main():
    cli = LightningCLI(
        SSLMOSLightningModule, MOSDataModule, save_config_kwargs={"overwrite": True}
    )


if __name__ == "__main__":
    cli_main()
