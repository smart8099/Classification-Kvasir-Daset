from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib


def inspect_folder(folder: Path) -> Counter:
    counts: Counter = Counter()
    files = sorted(folder.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No .nii.gz files found in {folder}")

    first = files[0]
    img = nib.load(str(first))
    print(f"\nFirst file in {folder.name}: {first.name}")
    print("shape:", img.shape)
    print("axcodes:", nib.aff2axcodes(img.affine))
    print("affine:\n", img.affine)

    for f in files:
        ax = nib.aff2axcodes(nib.load(str(f)).affine)
        counts[ax] += 1

    print(f"\nOrientation counts in {folder.name}:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-root",
        default=(
            "/Users/abdulbasit/Documents/phd_lifetime/endo_agent_project/"
            "DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task002_Synapse"
        ),
        help="Path to Task002_Synapse (contains imagesTr/imagesTs/labelsTr)",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also inspect imagesTs",
    )
    args = parser.parse_args()

    task_root = Path(args.task_root).expanduser().resolve()
    images_tr = task_root / "imagesTr"
    images_ts = task_root / "imagesTs"

    print("Task root:", task_root)
    inspect_folder(images_tr)

    if args.include_test:
        inspect_folder(images_ts)


if __name__ == "__main__":
    main()
