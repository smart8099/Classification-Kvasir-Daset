from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import models
import yaml


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="outputs/best.pth")
    parser.add_argument("--out", default="outputs/afacnet_cls.pth")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    labels_path = Path(cfg["output_dir"]) / "labels.json"
    if labels_path.exists():
        import json
        with open(labels_path, "r") as f:
            labels = json.load(f)["labels"]
        num_classes = len(labels)
    else:
        num_classes = 4

    state = torch.load(args.weights, map_location="cpu")
    # Infer num_classes from checkpoint
    fc_weight = state.get("fc.weight")
    inferred_classes = fc_weight.shape[0] if fc_weight is not None else num_classes
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, inferred_classes)
    model.load_state_dict(state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
