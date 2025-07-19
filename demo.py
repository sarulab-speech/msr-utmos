import argparse
import sys
from pathlib import Path

import gradio as gr
import torch
import torchaudio

# Add src to path to import sfi_utmos
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from sfi_utmos.model.ssl_mos import SSLMOSLightningModule

# Global variable for the model
model: SSLMOSLightningModule | None = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str):
    """Loads the model from the given checkpoint path."""
    global model
    model = SSLMOSLightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.eval()
    print(f"Model loaded from {checkpoint_path}")


def predict_mos(audio_path: str):
    """Predicts the MOS score for the given audio file."""
    if model is None:
        return "Error: Model not loaded. Please provide a valid checkpoint path."
    ratings = []
    for listner in range(1, 11):
        wav, sr = torchaudio.load(audio_path)
        if sr not in model.sr2id.keys():
            return f"Error: Sample rate {sr} not supported by the model. Supported rates: {list(model.sr2id.keys())}"
        waves = [wav.view(-1).to(model.device)]
        srs = torch.tensor(sr).view(1, -1).to(model.device)
        if model.condition_sr:
            srs = torch.stack(
                [torch.tensor(model.sr2id[sr.detach().cpu().item()]) for sr in srs]
            ).to(model.device)
        listner_tensor = torch.tensor(listner).view(-1).to(model.device)
        if hasattr(model, "is_sfi") and model.is_sfi:
            model.ssl_model.set_sample_rate(srs[0].item())
            waves = torch.nn.utils.rnn.pad_sequence(
                [w.view(-1) for w in waves], batch_first=True
            ).to(device)
        else:
            waves = [torchaudio.functional.resample(w, sr, 16_000) for w in waves]
        output = model.forward(
            waves,
            listner_tensor,
            srs,
        )
        ratings.append(output.cpu().item())
    mos_score = sum(ratings) / len(ratings)

    return f"{mos_score:.3f}"


def main():
    parser = argparse.ArgumentParser(description="Run MOS prediction demo with Gradio.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="model.ckpt",
        help="Path to the model checkpoint (.ckpt file).",
    )
    args = parser.parse_args()

    load_model(args.checkpoint_path)

    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)

    # Gradio interface
    iface = gr.Interface(
        fn=predict_mos,
        inputs=gr.Audio(type="filepath", label="Upload Audio File"),
        outputs="text",
        title="SFI-UTMOS: MOS Prediction Demo",
        description=(
            "Upload an audio file (WAV, MP3, etc.) to get its predicted Mean Opinion Score (MOS). "
        ),
    )
    iface.launch()


if __name__ == "__main__":
    main()
