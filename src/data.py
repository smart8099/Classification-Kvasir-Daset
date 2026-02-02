from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitFiles:
    train: List[Tuple[str, int]]
    val: List[Tuple[str, int]]
    test: List[Tuple[str, int]]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def find_class_dirs(data_root: Path) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    # If the root contains a single folder (e.g., kvasir-dataset-v3), dive in.
    children = [p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("__")]
    if len(children) == 1:
        data_root = children[0]

    class_dirs = [p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith("__")]
    return class_dirs


def build_class_map(data_root: Path, class_map: Dict[str, List[str]]) -> Dict[str, List[Path]]:
    class_dirs = find_class_dirs(data_root)
    name_to_dir = {p.name: p for p in class_dirs}

    if class_map:
        resolved: Dict[str, List[Path]] = {}
        for label, folders in class_map.items():
            resolved[label] = []
            for f in folders:
                if f not in name_to_dir:
                    raise ValueError(f"Folder '{f}' not found under {data_root}")
                resolved[label].append(name_to_dir[f])
        return resolved

    # If no class_map provided, treat each folder as a class label
    return {p.name: [p] for p in class_dirs}


def index_images(class_dirs: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
    indexed: Dict[str, List[Path]] = {}
    for label, dirs in class_dirs.items():
        files: List[Path] = []
        for d in dirs:
            for p in d.rglob("*"):
                if p.is_file() and _is_image(p):
                    files.append(p)
        files.sort()
        indexed[label] = files
    return indexed


def stratified_split(indexed: Dict[str, List[Path]], seed: int, split: Dict[str, float]) -> SplitFiles:
    rng = random.Random(seed)
    train: List[Tuple[str, int]] = []
    val: List[Tuple[str, int]] = []
    test: List[Tuple[str, int]] = []

    labels = sorted(indexed.keys())
    label_to_id = {label: i for i, label in enumerate(labels)}

    for label in labels:
        files = indexed[label][:]
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * split["train"])
        n_val = int(n * split["val"])
        n_test = n - n_train - n_val

        label_id = label_to_id[label]
        train.extend([(str(p), label_id) for p in files[:n_train]])
        val.extend([(str(p), label_id) for p in files[n_train:n_train + n_val]])
        test.extend([(str(p), label_id) for p in files[n_train + n_val:]])

        if n_test < 0:
            raise ValueError(f"Invalid split ratios for class {label}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return SplitFiles(train=train, val=val, test=test)


class ImageFolderList:
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
