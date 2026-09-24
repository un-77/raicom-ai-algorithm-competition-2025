"""Feature extraction, dataset handling, and Conformer model for SER."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torchaudio.models import Conformer


LABELS = ("anger", "fear", "happy", "neutral", "sad")


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 44_100
    n_mfcc: int = 10
    n_fft: int = 2_048
    win_length: int = 2_048
    hop_length: int = 512
    n_mels: int = 64
    max_frames: int = 300


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.size(0) > 1:
        return audio.mean(dim=0, keepdim=True)
    return audio


def augment_audio(audio: torch.Tensor) -> torch.Tensor:
    """Apply the lightweight augmentations used during the competition."""
    if random.random() < 0.5:
        audio = audio + 0.005 * torch.randn_like(audio)
    if random.random() < 0.3:
        limit = int(0.1 * audio.shape[-1])
        audio = torch.roll(audio, shifts=random.randint(-limit, limit), dims=-1)
    if random.random() < 0.3:
        audio = audio * random.uniform(0.8, 1.2)
    if random.random() < 0.3:
        audio = audio + 0.01 * torch.randn_like(audio)
    return audio.clamp(-1.0, 1.0)


def extract_features(
    audio: torch.Tensor,
    sample_rate: int,
    config: FeatureConfig = FeatureConfig(),
    augment: bool = False,
) -> torch.Tensor:
    """Return a fixed-length tensor shaped ``[time, 31]``."""
    audio = _mono(audio).float()
    if sample_rate != config.sample_rate:
        audio = T.Resample(sample_rate, config.sample_rate)(audio)
    if augment:
        audio = augment_audio(audio)

    mfcc = T.MFCC(
        sample_rate=config.sample_rate,
        n_mfcc=config.n_mfcc,
        melkwargs={
            "n_fft": config.n_fft,
            "win_length": config.win_length,
            "hop_length": config.hop_length,
            "n_mels": config.n_mels,
        },
    )(audio).squeeze(0)

    window = torch.hann_window(config.win_length, device=audio.device)
    stft = torch.stft(
        audio,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        window=window,
        return_complex=True,
    )
    magnitude = stft.abs().squeeze(0)
    frequencies = torch.linspace(
        0,
        config.sample_rate / 2,
        magnitude.size(0),
        device=audio.device,
    ).unsqueeze(1)
    denominator = magnitude.sum(dim=0) + 1e-10
    centroid = ((frequencies * magnitude).sum(dim=0) / denominator).unsqueeze(0)
    bandwidth = (
        (magnitude * (frequencies - centroid).abs()).sum(dim=0) / denominator
    ).unsqueeze(0)

    audio_np = audio.squeeze(0).detach().cpu().numpy()
    chroma = torch.from_numpy(
        librosa.feature.chroma_stft(
            y=audio_np,
            sr=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
    ).to(audio.device)
    contrast = torch.from_numpy(
        librosa.feature.spectral_contrast(
            y=audio_np,
            sr=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
    ).to(audio.device)

    frame_count = min(
        part.size(1) for part in (mfcc, centroid, bandwidth, chroma, contrast)
    )
    features = torch.cat(
        [part[:, :frame_count] for part in (mfcc, centroid, bandwidth, chroma, contrast)],
        dim=0,
    ).float()

    # Per-utterance normalization keeps training and inference self-contained.
    features = (features - features.mean(dim=1, keepdim=True)) / (
        features.std(dim=1, keepdim=True) + 1e-6
    )
    if frame_count < config.max_frames:
        features = F.pad(features, (0, config.max_frames - frame_count))
    else:
        features = features[:, : config.max_frames]
    return features.transpose(0, 1).contiguous()


class AudioEmotionDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        config: FeatureConfig = FeatureConfig(),
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.config = config
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        for class_index, label in enumerate(LABELS):
            class_dir = self.root_dir / label
            if not class_dir.is_dir():
                continue
            self.samples.extend(
                (path, class_index) for path in sorted(class_dir.glob("*.wav"))
            )
        if not self.samples:
            raise FileNotFoundError(
                f"No WAV files found under {self.root_dir}. Expected class folders: "
                + ", ".join(LABELS)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        audio, sample_rate = torchaudio.load(path)
        features = extract_features(
            audio,
            sample_rate,
            config=self.config,
            augment=self.augment,
        )
        return features, torch.tensor(label, dtype=torch.long)


class SERConformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 31,
        num_classes: int = len(LABELS),
        d_model: int = 144,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.fc_in = nn.Linear(32 * input_dim, d_model)
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=4,
            ffn_dim=256,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = inputs.shape
        lengths = torch.full(
            (batch_size,),
            time_steps,
            dtype=torch.long,
            device=inputs.device,
        )
        outputs = self.conv(inputs.unsqueeze(1))
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        outputs = outputs.view(batch_size, time_steps, -1)
        outputs = self.fc_in(outputs)
        outputs, _ = self.conformer(outputs, lengths)
        return self.classifier(outputs.mean(dim=1))
