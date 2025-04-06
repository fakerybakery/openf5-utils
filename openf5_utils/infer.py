"""
openf5-utils
Copyright (c) 2025 mrfakename. All rights reserved.
"""

import click
from pathlib import Path
import re
from huggingface_hub import HfApi, hf_hub_download
import soundfile as sf


@click.command()
@click.argument("repo_id", type=str)
@click.argument("ref_audio", type=click.Path(exists=True))
@click.argument("text", type=str)
def infer(repo_id, ref_audio, text):
    """
    Generate speech using F5-TTS model.

    REPO_ID: Hugging Face repository ID for the model
    REF_AUDIO: Path to reference audio file
    TEXT: Text to synthesize
    """
    click.echo(f"Model: {repo_id}")
    click.echo(f"Reference audio: {ref_audio}")
    click.echo(f"Text to synthesize: {text}")

    # Get the latest model checkpoint from HuggingFace Hub
    api = HfApi()
    files = api.list_repo_files(repo_id)

    # Find all model checkpoint files
    model_files = [f for f in files if re.match(r"model_\d+\.pt", f)]

    if not model_files:
        raise click.ClickException(f"No model checkpoints found in {repo_id}")

    # Extract model numbers and find the highest one
    model_numbers = [
        int(re.search(r"model_(\d+)\.pt", f).group(1)) for f in model_files
    ]
    latest_model_number = max(model_numbers)
    latest_model_file = f"model_{latest_model_number}.pt"

    click.echo(f"Using latest model checkpoint: {latest_model_file}")

    # Download the model file
    model_path = hf_hub_download(repo_id=repo_id, filename=latest_model_file)
    vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.json")

    from f5_tts.api import F5TTS

    model = F5TTS(ckpt_file=model_path, vocab_file=vocab_path)

    # Load reference audio
    ref_audio_path = Path(ref_audio)
    if not ref_audio_path.exists():
        raise click.ClickException(f"Reference audio file not found: {ref_audio}")

    wav, sr, _ = model.infer(
        ref_file=ref_audio_path, ref_text="", gen_text=text, seed=-1
    )

    # Save the generated audio
    output_path = Path("output.wav")
    sf.write(output_path, wav, sr)
    click.echo(f"Generated audio saved to {output_path}")


if __name__ == "__main__":
    infer()
