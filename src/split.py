from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data import build_class_map, index_images, stratified_split


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_split(output_dir: Path, split):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", split.train), ("val", split.val), ("test", split.test)]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(items, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_root = Path(cfg["data_root"]).resolve()
    class_map = cfg.get("class_map", {}) or {}

    class_dirs = build_class_map(data_root, class_map)
    indexed = index_images(class_dirs)
    split = stratified_split(indexed, seed=cfg["seed"], split=cfg["split"])

    split_dir = Path(cfg["output_dir"]) / "splits"
    save_split(split_dir, split)

    print(f"saved: {split_dir}/train.json")
    print(f"saved: {split_dir}/val.json")
    print(f"saved: {split_dir}/test.json")


if __name__ == "__main__":
    main()
