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
    if (split_dir / "train.json").exists():
        train_items, val_items, test_items = load_split(split_dir)
    else:
        split = stratified_split(indexed, seed=seed, split=cfg["split"])
        save_split(split_dir, split)
        train_items, val_items, test_items = split.train, split.val, split.test

    labels = sorted(class_dirs.keys())
    num_classes = len(labels)
    label_map_path = Path(cfg["output_dir"]) / "labels.json"
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump({"labels": labels}, f, indent=2)

    tf_train = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ])

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

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = 0.0
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
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)
        val_acc = correct / max(total, 1)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), output_dir / "best.pth")

        print(f"epoch={epoch+1} train_loss={running/ max(len(train_loader),1):.4f} val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), output_dir / "last.pth")


if __name__ == "__main__":
    main()
