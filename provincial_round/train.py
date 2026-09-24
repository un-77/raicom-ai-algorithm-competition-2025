"""Train the provincial-round speech emotion recognition model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import AudioEmotionDataset, LABELS, SERConformer, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root class-folder path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/ser_conformer_best.pth"),
        help="Best-checkpoint output path",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_epoch(
    model: SERConformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in tqdm(loader, leave=False):
            features = features.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            losses.append(loss.item())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    accuracy = float((np.asarray(predictions) == np.asarray(targets)).mean())
    weighted_f1 = f1_score(targets, predictions, average="weighted")
    return float(np.mean(losses)), accuracy, weighted_f1


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = AudioEmotionDataset(args.data, augment=True)
    val_dataset = AudioEmotionDataset(args.data, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    val_size = max(1, int(len(indices) * args.val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        raise ValueError("Dataset is too small for the requested validation split")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), shuffle=False, **loader_options
    )

    model = SERConformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy, val_f1 = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_f1)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_accuracy:.4f} f1={train_f1:.4f} | "
            f"val loss={val_loss:.4f} acc={val_accuracy:.4f} f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": LABELS,
                    "validation_f1": best_f1,
                    "seed": args.seed,
                },
                args.output,
            )
            print(f"Saved best checkpoint to {args.output} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
