from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict
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


def _case_id(path: Path) -> str:
    """Extract case ID from filename: numeric prefix for ERCPMP, full stem for Kvasir UUIDs."""
    m = re.match(r"(\d+)", path.name)
    return m.group(1) if m else path.stem


def stratified_split(indexed: Dict[str, List[Path]], seed: int, split: Dict[str, float]) -> SplitFiles:
    """Case-aware stratified split: all images from the same case stay in one split."""
    rng = random.Random(seed)
    train: List[Tuple[str, int]] = []
    val: List[Tuple[str, int]] = []
    test: List[Tuple[str, int]] = []

    labels = sorted(indexed.keys())
    label_to_id = {label: i for i, label in enumerate(labels)}

    for label in labels:
        # Group files by case
        cases: Dict[str, List[Path]] = defaultdict(list)
        for p in indexed[label]:
            cases[_case_id(p)].append(p)

        case_ids = list(cases.keys())
        rng.shuffle(case_ids)

        n = len(case_ids)
        n_train = max(1, int(n * split["train"]))
        n_val = max(1, int(n * split["val"])) if n > 2 else 0
        # Ensure we don't exceed total
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train_ids = case_ids[:n_train]
        val_ids = case_ids[n_train:n_train + n_val]
        test_ids = case_ids[n_train + n_val:]

        label_id = label_to_id[label]
        for cid in train_ids:
            train.extend([(str(p), label_id) for p in cases[cid]])
        for cid in val_ids:
            val.extend([(str(p), label_id) for p in cases[cid]])
        for cid in test_ids:
            test.extend([(str(p), label_id) for p in cases[cid]])

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
