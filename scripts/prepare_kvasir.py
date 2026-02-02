from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    # Return the extracted root folder if present
    children = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to kvasir-dataset-v3.zip")
    parser.add_argument("--out", default="./data/kvasir-dataset-v3", help="Extraction directory")
    args = parser.parse_args()

    zip_path = Path(args.zip).resolve()
    out_dir = Path(args.out).resolve()

    root = extract_zip(zip_path, out_dir)
    print(f"extracted to: {root}")


if __name__ == "__main__":
    main()
