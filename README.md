# Classification Model Project

This repo trains a plain PyTorch image classifier for endoscopy images.

## Data
- Place the Kvasir zip at `./data/kvasir-dataset-v3.zip`
- Extract it:
  ```bash
  python3 scripts/prepare_kvasir.py --zip ./data/kvasir-dataset-v3.zip --out ./data/kvasir-dataset-v3
  ```

## Config
Edit `configs/default.yaml`:
- `data_root`: path to the extracted dataset
- `class_map`: map dataset folders to labels

Example (2-class):
```yaml
class_map:
  Normal: ["normal-cecum", "normal-pylorus", "normal-z-line"]
  Polyp: ["polyps"]
```

## Train
```bash
python3 src/train.py --config configs/default.yaml
```

## Eval
```bash
python3 src/eval.py --config configs/default.yaml --weights outputs/best.pth
```

## Export
```bash
python3 src/export.py --config configs/default.yaml --weights outputs/best.pth --out outputs/afacnet_cls.pth
```
