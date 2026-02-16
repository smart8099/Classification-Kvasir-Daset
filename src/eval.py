from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import yaml

from data import ImageFolderList, MultiLabelImageList


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
    task_type = (cfg.get("task_type", "multiclass") or "multiclass").strip().lower()

    split_dir = Path(cfg["output_dir"]) / "splits"
    _, val_items, test_items = load_split(split_dir)
    if len(test_items) == 0:
        print("warning: test split is empty; evaluating on val split instead")
        test_items = val_items

    labels_path = Path(cfg["output_dir"]) / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"missing labels file: {labels_path}")
    with open(labels_path, "r") as f:
        labels = json.load(f)["labels"]

    norm_cfg = cfg.get("normalization", {}) or {}
    norm_enabled = bool(norm_cfg.get("enabled", False))
    stats_path = Path(norm_cfg.get("stats_path") or (Path(cfg["output_dir"]) / "normalize.json"))

    tf_eval_parts = [
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ]
    if norm_enabled:
        if not stats_path.exists():
            raise FileNotFoundError(f"normalization enabled but stats not found: {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        tf_eval_parts.append(transforms.Normalize(mean=stats["mean"], std=stats["std"]))
    tf_eval = transforms.Compose(tf_eval_parts)

    if task_type == "multiclass":
        test_ds = ImageFolderList(test_items, transform=tf_eval)
    else:
        test_ds = MultiLabelImageList(test_items, transform=tf_eval)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(labels)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    if task_type == "multiclass":
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels_batch in test_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)
        acc = correct / max(total, 1)
        print(f"test_acc={acc:.4f}")
    else:
        ml_cfg = cfg.get("multilabel", {}) or {}
        threshold = float(ml_cfg.get("threshold", 0.5))

        tp = torch.zeros(num_classes, dtype=torch.float64)
        fp = torch.zeros(num_classes, dtype=torch.float64)
        fn = torch.zeros(num_classes, dtype=torch.float64)

        with torch.no_grad():
            for images, labels_batch in test_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device).float()
                logits = model(images)
                preds = (torch.sigmoid(logits) >= threshold).float()

                tp += (preds * labels_batch).sum(dim=0).cpu().double()
                fp += (preds * (1.0 - labels_batch)).sum(dim=0).cpu().double()
                fn += ((1.0 - preds) * labels_batch).sum(dim=0).cpu().double()

        precision = tp / torch.clamp(tp + fp, min=1e-8)
        recall = tp / torch.clamp(tp + fn, min=1e-8)
        f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-8)

        tp_micro = tp.sum().item()
        fp_micro = fp.sum().item()
        fn_micro = fn.sum().item()
        p_micro = tp_micro / max(tp_micro + fp_micro, 1e-8)
        r_micro = tp_micro / max(tp_micro + fn_micro, 1e-8)
        f1_micro = (2.0 * p_micro * r_micro) / max(p_micro + r_micro, 1e-8)

        p_macro = precision.mean().item()
        r_macro = recall.mean().item()
        f1_macro = f1.mean().item()

        print(f"test_micro_f1={f1_micro:.4f}")
        print(f"test_macro_f1={f1_macro:.4f}")
        print(f"test_macro_precision={p_macro:.4f}")
        print(f"test_macro_recall={r_macro:.4f}")
        print("per_class_f1:")
        for i, name in enumerate(labels):
            print(f"  {name}: {f1[i].item():.4f}")


if __name__ == "__main__":
    main()
