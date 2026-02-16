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


def collect_logits_labels(model, loader, device):
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            logits = model(images).cpu()
            logits_all.append(logits)
            labels_all.append(labels_batch.float().cpu())
    if not logits_all:
        return torch.empty((0, 0)), torch.empty((0, 0))
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def multilabel_counts_from_preds(preds: torch.Tensor, labels: torch.Tensor):
    tp = (preds * labels).sum(dim=0).double()
    fp = (preds * (1.0 - labels)).sum(dim=0).double()
    fn = ((1.0 - preds) * labels).sum(dim=0).double()
    return tp, fp, fn


def multilabel_metrics_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
    precision = tp / torch.clamp(tp + fp, min=1e-8)
    recall = tp / torch.clamp(tp + fn, min=1e-8)
    f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1e-8)

    tp_micro = tp.sum().item()
    fp_micro = fp.sum().item()
    fn_micro = fn.sum().item()
    p_micro = tp_micro / max(tp_micro + fp_micro, 1e-8)
    r_micro = tp_micro / max(tp_micro + fn_micro, 1e-8)
    f1_micro = (2.0 * p_micro * r_micro) / max(p_micro + r_micro, 1e-8)

    return {
        "micro_f1": f1_micro,
        "macro_f1": f1.mean().item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "per_class_f1": f1,
    }


def tune_per_class_thresholds(logits: torch.Tensor, labels: torch.Tensor):
    probs = torch.sigmoid(logits)
    num_classes = probs.size(1)
    candidates = torch.linspace(0.1, 0.9, steps=17)
    best_thresholds = torch.full((num_classes,), 0.5, dtype=torch.float32)

    for c in range(num_classes):
        y = labels[:, c]
        p = probs[:, c]
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            pred = (p >= t).float()
            tp = (pred * y).sum().item()
            fp = (pred * (1.0 - y)).sum().item()
            fn = ((1.0 - pred) * y).sum().item()
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t.item())
        best_thresholds[c] = best_t

    return best_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="outputs/best.pth")
    parser.add_argument(
        "--tune-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For multilabel: tune per-class thresholds on val split before test evaluation.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    task_type = (cfg.get("task_type", "multiclass") or "multiclass").strip().lower()

    split_dir = Path(cfg["output_dir"]) / "splits"
    _, val_items, test_items = load_split(split_dir)
    has_test_split = len(test_items) > 0
    if not has_test_split:
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
        default_threshold = float(ml_cfg.get("threshold", 0.5))
        thresholds = torch.full((num_classes,), default_threshold, dtype=torch.float32)

        if args.tune_thresholds:
            val_ds = MultiLabelImageList(val_items, transform=tf_eval)
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg["num_workers"],
                pin_memory=True,
            )
            val_logits, val_labels = collect_logits_labels(model, val_loader, device)
            if val_logits.numel() > 0 and val_logits.shape[1] == num_classes:
                thresholds = tune_per_class_thresholds(val_logits, val_labels)
                print("thresholds_tuned_on=val")
                for i, name in enumerate(labels):
                    print(f"  {name}: {thresholds[i].item():.2f}")
                if not has_test_split:
                    print("warning: tuned and evaluated on val (test split is empty)")
            else:
                print("warning: could not tune thresholds from val split; using config threshold")

        test_logits, test_labels = collect_logits_labels(model, test_loader, device)
        test_probs = torch.sigmoid(test_logits)
        preds = (test_probs >= thresholds.unsqueeze(0)).float()
        tp, fp, fn = multilabel_counts_from_preds(preds, test_labels)
        metrics = multilabel_metrics_from_counts(tp, fp, fn)

        print(f"test_micro_f1={metrics['micro_f1']:.4f}")
        print(f"test_macro_f1={metrics['macro_f1']:.4f}")
        print(f"test_macro_precision={metrics['macro_precision']:.4f}")
        print(f"test_macro_recall={metrics['macro_recall']:.4f}")
        print("per_class_f1:")
        for i, name in enumerate(labels):
            print(f"  {name}: {metrics['per_class_f1'][i].item():.4f}")


if __name__ == "__main__":
    main()
