from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import yaml

from data import (
    build_class_map,
    index_images,
    stratified_split,
    ImageFolderList,
    MultiLabelImageList,
)


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


def compute_channel_stats(ds, batch_size: int, num_workers: int):
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


def ensure_labels_file(output_dir: Path, labels):
    labels_path = output_dir / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump({"labels": labels}, f, indent=2)


def build_multiclass_sample_weights(train_items, num_classes: int, exponent: float = 1.0):
    label_counts = Counter(label for _, label in train_items)
    total = max(len(train_items), 1)
    class_weights = torch.tensor(
        [(total / max(label_counts.get(i, 1), 1)) ** exponent for i in range(num_classes)],
        dtype=torch.double,
    )
    return torch.tensor([class_weights[label].item() for _, label in train_items], dtype=torch.double)


def build_multilabel_sample_weights(train_items, num_classes: int, exponent: float = 1.0):
    pos_counts = torch.zeros(num_classes, dtype=torch.float64)
    all_zero_count = 0
    labels_list = []
    for item in train_items:
        y = item["labels"] if isinstance(item, dict) else item[1]
        y_t = torch.tensor(y, dtype=torch.float64)
        labels_list.append(y_t)
        pos_counts += y_t
        if y_t.sum().item() == 0.0:
            all_zero_count += 1

    total = max(len(train_items), 1)
    inv_pos = (total / torch.clamp(pos_counts, min=1.0)) ** exponent
    inv_all_zero = float((total / max(all_zero_count, 1)) ** exponent)

    sample_weights = []
    for y_t in labels_list:
        pos_idx = y_t > 0.0
        if pos_idx.any():
            sample_weights.append(inv_pos[pos_idx].mean().item())
        else:
            sample_weights.append(inv_all_zero)
    return torch.tensor(sample_weights, dtype=torch.double)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg["seed"])
    torch.manual_seed(seed)

    task_type = (cfg.get("task_type", "multiclass") or "multiclass").strip().lower()
    if task_type not in {"multiclass", "multilabel"}:
        raise ValueError("task_type must be 'multiclass' or 'multilabel'")

    split_dir = Path(cfg["output_dir"]) / "splits"
    split_exists = (split_dir / "train.json").exists()

    labels = None
    num_classes = None

    if task_type == "multiclass":
        data_root = Path(cfg["data_root"]).resolve()
        class_map = cfg.get("class_map", {}) or {}
        class_dirs = build_class_map(data_root, class_map)

        if split_exists:
            train_items, val_items, test_items = load_split(split_dir)
            labels_path = Path(cfg["output_dir"]) / "labels.json"
            if not labels_path.exists():
                labels = sorted(class_dirs.keys())
                ensure_labels_file(Path(cfg["output_dir"]), labels)
            else:
                with open(labels_path, "r") as f:
                    labels = json.load(f)["labels"]
        else:
            indexed = index_images(class_dirs)
            split = stratified_split(indexed, seed=seed, split=cfg["split"])
            save_split(split_dir, split)
            train_items, val_items, test_items = split.train, split.val, split.test
            labels = sorted(class_dirs.keys())
            ensure_labels_file(Path(cfg["output_dir"]), labels)

        num_classes = len(labels)
    else:
        if not split_exists:
            raise FileNotFoundError(
                f"multilabel training expects precomputed splits at {split_dir}. "
                "Run scripts/prepare_synapse_classification.py first."
            )
        train_items, val_items, test_items = load_split(split_dir)
        labels_path = Path(cfg["output_dir"]) / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r") as f:
                labels = json.load(f)["labels"]
            num_classes = len(labels)
        else:
            if not train_items:
                raise ValueError("empty train split; cannot infer class count")
            first = train_items[0]
            if isinstance(first, dict):
                num_classes = len(first["labels"])
            else:
                num_classes = len(first[1])
            labels = [f"class_{i}" for i in range(num_classes)]
            ensure_labels_file(Path(cfg["output_dir"]), labels)

    norm_cfg = cfg.get("normalization", {}) or {}
    norm_enabled = bool(norm_cfg.get("enabled", False))
    stats_path = Path(norm_cfg.get("stats_path") or (Path(cfg["output_dir"]) / "normalize.json"))

    aug = cfg.get("augmentation", {}) or {}
    color = aug.get("color_jitter", {}) or {}
    tf_train_parts = [transforms.Resize((cfg["img_size"], cfg["img_size"]))]
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

    tf_eval_parts = [
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ]

    if norm_enabled:
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            norm_mean = stats["mean"]
            norm_std = stats["std"]
        else:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            tf_stats = transforms.Compose(
                [
                    transforms.Resize((cfg["img_size"], cfg["img_size"])),
                    transforms.ToTensor(),
                ]
            )
            if task_type == "multiclass":
                ds_stats = ImageFolderList(train_items, transform=tf_stats)
            else:
                ds_stats = MultiLabelImageList(train_items, transform=tf_stats)
            norm_mean, norm_std = compute_channel_stats(
                ds_stats,
                batch_size=cfg["batch_size"],
                num_workers=cfg["num_workers"],
            )
            with open(stats_path, "w") as f:
                json.dump({"mean": norm_mean, "std": norm_std}, f, indent=2)
            print(f"saved normalization stats: {stats_path}")

        tf_train_parts.append(transforms.Normalize(mean=norm_mean, std=norm_std))
        tf_eval_parts.append(transforms.Normalize(mean=norm_mean, std=norm_std))

    tf_train = transforms.Compose(tf_train_parts)
    tf_eval = transforms.Compose(tf_eval_parts)

    if task_type == "multiclass":
        train_ds = ImageFolderList(train_items, transform=tf_train)
        val_ds = ImageFolderList(val_items, transform=tf_eval)
    else:
        train_ds = MultiLabelImageList(train_items, transform=tf_train)
        val_ds = MultiLabelImageList(val_items, transform=tf_eval)

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

    if task_type == "multiclass":
        label_counts = Counter(label for _, label in train_items)
        total_train = len(train_items)
        class_weights = torch.tensor(
            [total_train / max(label_counts.get(i, 1), 1) for i in range(num_classes)],
            dtype=torch.float32,
            device=device,
        )
        class_weights = class_weights / class_weights.sum() * num_classes
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        pos_counts = torch.zeros(num_classes, dtype=torch.float32)
        for item in train_items:
            y = item["labels"] if isinstance(item, dict) else item[1]
            pos_counts += torch.tensor(y, dtype=torch.float32)
        n = max(float(len(train_items)), 1.0)
        neg_counts = n - pos_counts
        pos_weight = neg_counts / torch.clamp(pos_counts, min=1.0)
        ml_cfg = cfg.get("multilabel", {}) or {}
        pos_weight_power = float(ml_cfg.get("pos_weight_power", 1.0))
        pos_weight_min = float(ml_cfg.get("pos_weight_min", 1.0))
        pos_weight_max = ml_cfg.get("pos_weight_max", None)
        if pos_weight_power != 1.0:
            pos_weight = torch.pow(pos_weight, pos_weight_power)
        if pos_weight_max is None:
            pos_weight = torch.clamp(pos_weight, min=pos_weight_min)
        else:
            pos_weight = torch.clamp(pos_weight, min=pos_weight_min, max=float(pos_weight_max))
        print(
            "pos_weight_stats "
            f"min={pos_weight.min().item():.4f} "
            f"median={pos_weight.median().item():.4f} "
            f"max={pos_weight.max().item():.4f}"
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    imb_cfg = cfg.get("imbalance", {}) or {}
    sampler_cfg = imb_cfg.get("sampler", {}) or {}
    sampler_enabled = bool(sampler_cfg.get("enabled", False))
    sampler_replacement = bool(sampler_cfg.get("replacement", True))
    sampler_exponent = float(sampler_cfg.get("exponent", 1.0))
    train_sampler = None
    if sampler_enabled:
        if task_type == "multiclass":
            sample_weights = build_multiclass_sample_weights(
                train_items,
                num_classes=num_classes,
                exponent=sampler_exponent,
            )
        else:
            sample_weights = build_multilabel_sample_weights(
                train_items,
                num_classes=num_classes,
                exponent=sampler_exponent,
            )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=sampler_replacement,
        )
        print(
            "weighted_sampler enabled "
            f"replacement={sampler_replacement} exponent={sampler_exponent} "
            f"w_min={sample_weights.min().item():.4f} "
            f"w_max={sample_weights.max().item():.4f}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched_cfg = cfg.get("scheduler", {}) or {}
    sched_name = (sched_cfg.get("name", "none") or "none").strip().lower()
    scheduler = None
    if sched_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 2)),
            min_lr=float(sched_cfg.get("min_lr", 1.0e-6)),
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("t_max", cfg["epochs"])),
            eta_min=float(sched_cfg.get("eta_min", 1.0e-6)),
        )
    elif sched_name != "none":
        raise ValueError("scheduler.name must be one of: none, reduce_on_plateau, cosine")

    best_val = -1.0
    es_cfg = cfg.get("early_stopping", {}) or {}
    patience = int(es_cfg.get("patience", 0))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    no_improve = 0
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ml_cfg = cfg.get("multilabel", {}) or {}
    threshold = float(ml_cfg.get("threshold", 0.5))

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for images, labels_batch in train_loader:
            images = images.to(device)
            if task_type == "multiclass":
                labels_batch = labels_batch.to(device)
            else:
                labels_batch = labels_batch.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        val_running_loss = 0.0
        if task_type == "multiclass":
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images = images.to(device)
                    labels_batch = labels_batch.to(device)
                    outputs = model(images)
                    val_running_loss += criterion(outputs, labels_batch).item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels_batch).sum().item()
                    total += labels_batch.size(0)
            val_metric = correct / max(total, 1)
        else:
            tp = fp = fn = 0.0
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images = images.to(device)
                    labels_batch = labels_batch.to(device).float()
                    outputs = model(images)
                    val_running_loss += criterion(outputs, labels_batch).item()
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= threshold).float()
                    tp += (preds * labels_batch).sum().item()
                    fp += (preds * (1.0 - labels_batch)).sum().item()
                    fn += ((1.0 - preds) * labels_batch).sum().item()
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            val_metric = (2.0 * precision * recall) / max(precision + recall, 1e-8)

        avg_val_loss = val_running_loss / max(len(val_loader), 1)

        if val_metric > best_val + min_delta:
            best_val = val_metric
            torch.save(model.state_dict(), output_dir / "best.pth")
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            if sched_name == "reduce_on_plateau":
                scheduler.step(val_metric)
            else:
                scheduler.step()

        metric_name = "val_acc" if task_type == "multiclass" else "val_micro_f1"
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch+1} train_loss={running / max(len(train_loader),1):.4f} "
            f"val_loss={avg_val_loss:.4f} {metric_name}={val_metric:.4f} lr={current_lr:.6g}"
        )

        if patience > 0 and no_improve >= patience:
            print(f"early_stop: no improvement for {patience} epochs")
            break

    torch.save(model.state_dict(), output_dir / "last.pth")


if __name__ == "__main__":
    main()
