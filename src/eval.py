from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
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
    num_classes = len(labels)

    model = models.resnet18(weights=None)  # architecture only; weights loaded below
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    # --- torchmetrics collectors ---
    conf_mat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)

    macro_precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    macro_recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            preds = model(images).argmax(dim=1)

            conf_mat.update(preds, labels_batch)
            precision_metric.update(preds, labels_batch)
            recall_metric.update(preds, labels_batch)
            f1_metric.update(preds, labels_batch)
            acc_metric.update(preds, labels_batch)
            macro_precision.update(preds, labels_batch)
            macro_recall.update(preds, labels_batch)
            macro_f1.update(preds, labels_batch)

    # --- Print results ---
    cm = conf_mat.compute().cpu()
    prec = precision_metric.compute().cpu()
    rec = recall_metric.compute().cpu()
    f1 = f1_metric.compute().cpu()
    acc = acc_metric.compute().item()

    header = "".ljust(16) + "".join(l[:12].ljust(14) for l in labels)
    print("\n=== Confusion Matrix (rows=true, cols=predicted) ===")
    print(header)
    for i, label in enumerate(labels):
        row = label[:14].ljust(16) + "".join(str(int(cm[i, j].item())).ljust(14) for j in range(num_classes))
        print(row)

    print("\n=== Per-Class Metrics ===")
    print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 58)
    for i, label in enumerate(labels):
        support = int(cm[i, :].sum().item())
        print(f"{label:<16} {prec[i].item():>10.4f} {rec[i].item():>10.4f} {f1[i].item():>10.4f} {support:>10d}")

    mp = macro_precision.compute().item()
    mr = macro_recall.compute().item()
    mf = macro_f1.compute().item()
    total = int(cm.sum().item())
    print("-" * 58)
    print(f"{'Macro Avg':<16} {mp:>10.4f} {mr:>10.4f} {mf:>10.4f} {total:>10d}")
    print(f"\nOverall test_acc={acc:.4f}  ({int(acc * total)}/{total})")


if __name__ == "__main__":
    main()
