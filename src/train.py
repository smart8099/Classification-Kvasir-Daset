from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import yaml

from collections import Counter

from data import build_class_map, index_images, stratified_split, ImageFolderList


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_split(output_dir: Path, split):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", split.train), ("val", split.val), ("test", split.test)]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(items, f)


def load_split(output_dir: Path):
    def _load(name: str):
        with open(output_dir / f"{name}.json", "r") as f:
            return json.load(f)
    return _load("train"), _load("val"), _load("test")

def compute_channel_stats(items, img_size: int, batch_size: int, num_workers: int):
    tf_stats = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ds = ImageFolderList(items, transform=tf_stats)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    channel_sum = torch.zeros(3, dtype=torch.double)
    channel_sum_sq = torch.zeros(3, dtype=torch.double)
    num_pixels = 0
    for images, _ in loader:
        images = images.double()
        b, c, h, w = images.shape
        if c != 3:
            raise ValueError(f"expected 3 channels, got {c}")
        num_pixels += b * h * w
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
    if num_pixels == 0:
        raise ValueError("no pixels found while computing normalization stats")
    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sum_sq / num_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    seed = cfg["seed"]
    torch.manual_seed(seed)

    data_root = Path(cfg["data_root"]).resolve()
    class_map = cfg.get("class_map", {}) or {}

    class_dirs = build_class_map(data_root, class_map)
    indexed = index_images(class_dirs)

    split_dir = Path(cfg["output_dir"]) / "splits"
    # Always regenerate splits with case-aware logic
    for old in split_dir.glob("*.json"):
        old.unlink()
    split = stratified_split(indexed, seed=seed, split=cfg["split"])
    save_split(split_dir, split)
    train_items, val_items, test_items = split.train, split.val, split.test

    labels = sorted(class_dirs.keys())
    num_classes = len(labels)
    label_map_path = Path(cfg["output_dir"]) / "labels.json"
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump({"labels": labels}, f, indent=2)

    norm_cfg = cfg.get("normalization", {}) or {}
    norm_enabled = bool(norm_cfg.get("enabled", False))
    stats_path = Path(
        norm_cfg.get("stats_path") or (Path(cfg["output_dir"]) / "normalize.json")
    )
    norm_mean = None
    norm_std = None
    if norm_enabled:
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            norm_mean = stats["mean"]
            norm_std = stats["std"]
        else:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            norm_mean, norm_std = compute_channel_stats(
                train_items,
                img_size=cfg["img_size"],
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"],
            )
            with open(stats_path, "w") as f:
                json.dump({"mean": norm_mean, "std": norm_std}, f, indent=2)
            print(f"saved normalization stats: {stats_path}")

    aug = cfg.get("augmentation", {}) or {}
    color = aug.get("color_jitter", {}) or {}
    tf_train_parts = [
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
    ]
    if aug.get("hflip", True):
        tf_train_parts.append(transforms.RandomHorizontalFlip())
    if aug.get("vflip", False):
        tf_train_parts.append(transforms.RandomVerticalFlip())
    if aug.get("rotation_deg", 0) > 0:
        tf_train_parts.append(transforms.RandomRotation(aug.get("rotation_deg", 0)))
    if color:
        tf_train_parts.append(
            transforms.ColorJitter(
                brightness=color.get("brightness", 0.0),
                contrast=color.get("contrast", 0.0),
                saturation=color.get("saturation", 0.0),
                hue=color.get("hue", 0.0),
            )
        )
    tf_train_parts.append(transforms.ToTensor())
    if norm_enabled:
        tf_train_parts.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    tf_train = transforms.Compose(tf_train_parts)
    tf_eval_parts = [
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ]
    if norm_enabled:
        tf_eval_parts.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    tf_eval = transforms.Compose(tf_eval_parts)

    train_ds = ImageFolderList(train_items, transform=tf_train)
    val_ds = ImageFolderList(val_items, transform=tf_eval)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Class-weighted loss to handle severe imbalance (e.g. Cancer)
    label_counts = Counter(label for _, label in train_items)
    total_train = len(train_items)
    class_weights = torch.tensor(
        [total_train / max(label_counts.get(i, 1), 1) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalise
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = 0.0
    es_cfg = cfg.get("early_stopping", {}) or {}
    patience = int(es_cfg.get("patience", 0))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    no_improve = 0
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels_batch)
                val_running_loss += val_loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)
        val_acc = correct / max(total, 1)
        avg_val_loss = val_running_loss / max(len(val_loader), 1)

        if val_acc > best_val + min_delta:
            best_val = val_acc
            torch.save(model.state_dict(), output_dir / "best.pth")
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"epoch={epoch+1} train_loss={running/ max(len(train_loader),1):.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if patience > 0 and no_improve >= patience:
            print(f"early_stop: no improvement for {patience} epochs")
            break

    torch.save(model.state_dict(), output_dir / "last.pth")


if __name__ == "__main__":
    main()
