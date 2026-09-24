"""Predict one of five speech-emotion classes for a WAV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

from model import LABELS, SERConformer, extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    labels = tuple(checkpoint.get("labels", LABELS))

    model = SERConformer(num_classes=len(labels)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    audio, sample_rate = torchaudio.load(args.audio)
    features = extract_features(audio, sample_rate).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(features).softmax(dim=1)
    class_index = int(probabilities.argmax(dim=1).item())
    confidence = float(probabilities[0, class_index].item())
    print(f"label={labels[class_index]} confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
