from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import yaml

from data import ImageFolderList


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split(output_dir: Path):
    def _load(name: str):
        with open(output_dir / f"{name}.json", "r") as f:
            return json.load(f)
    return _load("train"), _load("val"), _load("test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="outputs/best.pth")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    output_dir = Path(cfg["output_dir"]) / "splits"
    _, _, test_items = load_split(output_dir)

    labels_path = Path(cfg["output_dir"]) / "labels.json"
    with open(labels_path, "r") as f:
        labels = json.load(f)["labels"]

    norm_cfg = cfg.get("normalization", {}) or {}
    norm_enabled = bool(norm_cfg.get("enabled", False))
    stats_path = Path(
        norm_cfg.get("stats_path") or (Path(cfg["output_dir"]) / "normalize.json")
    )
    norm_mean = None
    norm_std = None
    if norm_enabled:
        if not stats_path.exists():
            raise FileNotFoundError(
                f"normalization enabled but stats not found: {stats_path}"
            )
        with open(stats_path, "r") as f:
            stats = json.load(f)
        norm_mean = stats["mean"]
        norm_std = stats["std"]

    tf_eval_parts = [
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ]
    if norm_enabled:
        tf_eval_parts.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    tf_eval = transforms.Compose(tf_eval_parts)

    test_ds = ImageFolderList(test_items, transform=tf_eval)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

    acc = correct / max(total, 1)
    print(f"test_acc={acc:.4f}")


if __name__ == "__main__":
    main()
