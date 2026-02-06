from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from src.data import build_class_map, index_images, stratified_split, ImageFolderList


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split(output_dir: Path):
    def _load(name: str):
        with open(output_dir / f"{name}.json", "r") as f:
            return json.load(f)
    return _load("train"), _load("val"), _load("test")


def save_split(output_dir: Path, split):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", split.train), ("val", split.val), ("test", split.test)]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(items, f)


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
    data_root = Path(cfg["data_root"]).resolve()
    class_map = cfg.get("class_map", {}) or {}

    class_dirs = build_class_map(data_root, class_map)
    indexed = index_images(class_dirs)

    split_dir = Path(cfg["output_dir"]) / "splits"
    if (split_dir / "train.json").exists():
        train_items, _, _ = load_split(split_dir)
    else:
        split = stratified_split(indexed, seed=cfg["seed"], split=cfg["split"])
        save_split(split_dir, split)
        train_items = split.train

    norm_cfg = cfg.get("normalization", {}) or {}
    stats_path = Path(
        norm_cfg.get("stats_path") or (Path(cfg["output_dir"]) / "normalize.json")
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    mean, std = compute_channel_stats(
        train_items,
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    with open(stats_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    print(f"saved normalization stats: {stats_path}")
    print(f"mean={mean}")
    print(f"std={std}")


if __name__ == "__main__":
    main()
