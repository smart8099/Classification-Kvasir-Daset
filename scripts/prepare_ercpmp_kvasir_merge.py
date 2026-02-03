from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    run(["/usr/bin/unzip", "-q", str(zip_path), "-d", str(out_dir)])


def extract_rar(rar_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Prefer unar for RAR5 support on macOS
    if shutil.which("unar"):
        run(["unar", "-o", str(out_dir), str(rar_path)])
        return
    if shutil.which("unrar"):
        run(["unrar", "x", "-y", str(rar_path), str(out_dir)])
        return
    if shutil.which("7z"):
        run(["7z", "x", "-y", f"-o{out_dir}", str(rar_path)])
        return
    raise RuntimeError("Need unar/unrar/7z to extract .rar files. Install unar or p7zip.")


def find_kvasir_root(root: Path) -> Path:
    if (root / "kvasir-dataset-v3").exists():
        root = root / "kvasir-dataset-v3"
    # Some zips nest another kvasir-dataset-v3 folder
    if (root / "kvasir-dataset-v3").exists():
        root = root / "kvasir-dataset-v3"
    return root


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def parse_ercpmp_label(name: str) -> str | None:
    lower = name.lower()
    if "adenocarcinoma" in lower:
        return "Cancer"
    if "tubular" in lower or "tubulovillous" in lower or "villous" in lower:
        return "Adenoma"
    if "hyperplastic" in lower or "serrated" in lower:
        return "Polyp"
    if "jnet_3" in lower:
        return "Cancer"
    if "jnet_2a" in lower or "jnet_2b" in lower:
        return "Adenoma"
    if "jnet_1" in lower:
        return "Polyp"
    return None


def case_id_from_name(name: str) -> str:
    match = re.match(r"(\d+)", name)
    return match.group(1) if match else name


def extract_frames(video_path: Path, frames_dir: Path, fps: float) -> List[Path]:
    ensure_dir(frames_dir)
    out_pattern = frames_dir / f"{video_path.stem}_%06d.jpg"
    if any(frames_dir.glob(f"{video_path.stem}_*.jpg")):
        return list(frames_dir.glob(f"{video_path.stem}_*.jpg"))
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            str(out_pattern),
        ]
    )
    return list(frames_dir.glob(f"{video_path.stem}_*.jpg"))


def split_cases(
    items: List[Tuple[Path, str, str]], seed: int, split: Dict[str, float]
) -> Dict[str, List[Tuple[Path, str, str]]]:
    # items: (path, label, case_id)
    by_label_case: Dict[str, Dict[str, List[Tuple[Path, str, str]]]] = defaultdict(lambda: defaultdict(list))
    for path, label, case_id in items:
        by_label_case[label][case_id].append((path, label, case_id))

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for label, cases in by_label_case.items():
        case_ids = list(cases.keys())
        rng.shuffle(case_ids)
        n = len(case_ids)
        n_train = int(n * split["train"])
        n_val = int(n * split["val"])
        train_ids = set(case_ids[:n_train])
        val_ids = set(case_ids[n_train : n_train + n_val])
        test_ids = set(case_ids[n_train + n_val :])

        for cid in train_ids:
            splits["train"].extend(cases[cid])
        for cid in val_ids:
            splits["val"].extend(cases[cid])
        for cid in test_ids:
            splits["test"].extend(cases[cid])

    return splits


def split_kvasir(
    items: List[Tuple[Path, str, str]], seed: int, split: Dict[str, float]
) -> Dict[str, List[Tuple[Path, str, str]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Tuple[Path, str, str]]] = defaultdict(list)
    for item in items:
        by_label[item[1]].append(item)

    splits = {"train": [], "val": [], "test": []}
    for label, label_items in by_label.items():
        rng.shuffle(label_items)
        n = len(label_items)
        n_train = int(n * split["train"])
        n_val = int(n * split["val"])
        splits["train"].extend(label_items[:n_train])
        splits["val"].extend(label_items[n_train : n_train + n_val])
        splits["test"].extend(label_items[n_train + n_val :])
    return splits


def build_label_ids(labels: Iterable[str]) -> Dict[str, int]:
    labels_sorted = sorted(set(labels))
    return {label: i for i, label in enumerate(labels_sorted)}


def write_split_json(
    splits: Dict[str, List[Tuple[Path, str, str]]], label_to_id: Dict[str, int], out_dir: Path
) -> None:
    ensure_dir(out_dir)
    for split_name, items in splits.items():
        payload = [(str(path), label_to_id[label]) for path, label, _ in items]
        with open(out_dir / f"{split_name}.json", "w") as f:
            json.dump(payload, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvasir-zip")
    parser.add_argument("--ercpmp-rar")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split-out", default="outputs_4class/splits")
    parser.add_argument("--clean", action="store_true", help="Remove temporary _work directory after merge")
    parser.add_argument("--only-split", action="store_true", help="Skip extraction/merge and only create splits from out-root")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.8,0.1,0.1")
    args = parser.parse_args()

    if not args.only_split:
        if not args.kvasir_zip or not args.ercpmp_rar:
            raise SystemExit("--kvasir-zip and --ercpmp-rar are required unless --only-split is set")
        kvasir_zip = Path(args.kvasir_zip).expanduser().resolve()
        ercpmp_rar = Path(args.ercpmp_rar).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    work_dir = out_root / "_work"
    ensure_dir(work_dir)

    split_vals = [float(v.strip()) for v in args.split.split(",")]
    split_cfg = {"train": split_vals[0], "val": split_vals[1], "test": split_vals[2]}

    kvasir_items: List[Tuple[Path, str, str]] = []
    ercpmp_items: List[Tuple[Path, str, str]] = []

    if not args.only_split:
        # Extract
        kvasir_extract = work_dir / "kvasir"
        ercpmp_extract = work_dir / "ercpmp"
        if not kvasir_extract.exists():
            extract_zip(kvasir_zip, kvasir_extract)
        if not ercpmp_extract.exists():
            extract_rar(ercpmp_rar, ercpmp_extract)

        # Prepare output folders
        classes = ["Adenoma", "Cancer", "Normal", "Polyp"]
        for c in classes:
            ensure_dir(out_root / c)

        # Kvasir items
        kvasir_root = find_kvasir_root(kvasir_extract)
        kvasir_normals = ["normal-cecum", "normal-pylorus", "normal-z-line"]
        kvasir_polyp = ["polyps"]
        for folder in kvasir_normals:
            for img in (kvasir_root / folder).rglob("*"):
                if img.is_file() and is_image(img):
                    dst = out_root / "Normal" / img.name
                    safe_copy(img, dst)
                    kvasir_items.append((dst, "Normal", f"kvasir_{folder}"))
        for folder in kvasir_polyp:
            for img in (kvasir_root / folder).rglob("*"):
                if img.is_file() and is_image(img):
                    dst = out_root / "Polyp" / img.name
                    safe_copy(img, dst)
                    kvasir_items.append((dst, "Polyp", "kvasir_polyps"))

        # ERCPMP items (images + video frames)
        frames_root = work_dir / "frames"
        ensure_dir(frames_root)
        for path in ercpmp_extract.rglob("*"):
            if not path.is_file():
                continue
            label = parse_ercpmp_label(path.name)
            if label is None:
                continue
            case_id = case_id_from_name(path.name)
            if is_image(path):
                dst = out_root / label / path.name
                safe_copy(path, dst)
                ercpmp_items.append((dst, label, case_id))
            elif is_video(path):
                case_frames_dir = frames_root / case_id
                frames = extract_frames(path, case_frames_dir, fps=args.fps)
                for frame in frames:
                    dst = out_root / label / frame.name
                    safe_copy(frame, dst)
                    ercpmp_items.append((dst, label, case_id))
    else:
        # Only split based on existing out_root contents
        for label_dir in out_root.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img in label_dir.rglob("*"):
                if img.is_file() and is_image(img):
                    case_id = case_id_from_name(img.name)
                    # Treat all non-kvasir classes as ERCPMP for case-based splitting
                    if label == "Normal" or label == "Polyp":
                        kvasir_items.append((img, label, f"kvasir_{label}"))
                    else:
                        ercpmp_items.append((img, label, case_id))

    # Split: ERCPMP by case, Kvasir by image
    ercpmp_splits = split_cases(ercpmp_items, seed=args.seed, split=split_cfg) if ercpmp_items else {"train": [], "val": [], "test": []}
    kvasir_splits = split_kvasir(kvasir_items, seed=args.seed, split=split_cfg) if kvasir_items else {"train": [], "val": [], "test": []}

    merged_splits = {k: ercpmp_splits[k] + kvasir_splits[k] for k in ["train", "val", "test"]}
    label_to_id = build_label_ids([label for _, label, _ in kvasir_items + ercpmp_items])

    splits_dir = Path(args.split_out).expanduser().resolve()
    write_split_json(merged_splits, label_to_id, splits_dir)

    with open(out_root / "labels.json", "w") as f:
        json.dump({"labels": sorted(label_to_id.keys())}, f, indent=2)

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)

    print(
        f"done: kvasir={len(kvasir_items)} ercpmp={len(ercpmp_items)} "
        f"out={out_root} splits={splits_dir}"
    )


if __name__ == "__main__":
    main()
