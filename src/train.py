from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.digit_model import SudokuDigitCNN
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sudoku digit CNN.")
    parser.add_argument("--data-dir", default="data/processed/digits", help="Folder dataset root.")
    parser.add_argument("--output", default="models/digit_cnn.pt", help="Output model path.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_loaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    preds: list[int] = []
    targets: list[int] = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return sum(losses) / max(len(losses), 1), accuracy_score(targets, preds)


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.plot(history["train_acc"], label="train acc")
    plt.plot(history["val_acc"], label="val acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close()


@torch.no_grad()
def save_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, output_dir: Path) -> None:
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device))
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        targets.extend(labels.tolist())
    matrix = confusion_matrix(targets, preds, labels=list(range(10)))
    ConfusionMatrixDisplay(matrix, display_labels=list(range(10))).plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = ensure_dir("outputs/training")

    device = torch.device(args.device)
    train_loader, val_loader = build_loaders(data_dir, args.batch_size)
    model = SudokuDigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "class_to_idx": val_loader.dataset.class_to_idx}, output_path)

    plot_history(history, report_dir)
    save_confusion_matrix(model, val_loader, device, report_dir)
    print(f"Saved best model to {output_path}")


if __name__ == "__main__":
    main()
