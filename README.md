# Classification Model Project

This repo trains a plain PyTorch image classifier for endoscopy images.

## Data
- Place the Kvasir zip at `./data/kvasir-dataset-v3.zip`
- Extract it:
  ```bash
  python3 scripts/prepare_kvasir.py --zip ./data/kvasir-dataset-v3.zip --out ./data/kvasir-dataset-v3
  ```

### Synapse -> classification slices
Synapse is a 3D CT segmentation dataset, so first convert labeled volumes into 2D class-labeled slices:
```bash
python3 scripts/prepare_synapse_classification.py \
  --synapse-root /Users/abdulbasit/Documents/phd_lifetime/endo_agent_project/DATASET_Synapse \
  --task Task002_Synapse \
  --out-root data/synapse_classification \
  --split-out outputs_synapse/splits \
  --split 0.8,0.1,0.1 \
  --window 50,400 \
  --axis axial \
  --min-mask-pixels 200 \
  --slice-step 1
```

Train/val only + unlabeled `imagesTs` export (recommended flow):
```bash
python3 scripts/prepare_synapse_classification.py \
  --synapse-root /Users/abdulbasit/Documents/phd_lifetime/endo_agent_project/DATASET_Synapse \
  --task Task002_Synapse \
  --out-root data/synapse_classification_all_organs \
  --split-out outputs_synapse_all_organs/splits \
  --split 0.8,0.2 \
  --case-split-mode train-val-only \
  --window 50,400 \
  --axis axial \
  --slice-step 1 \
  --class-scheme all-organs \
  --label-policy presence \
  --presence-min-pixels 30 \
  --export-images-ts
```

True multilabel export (one slice path with multi-hot target vector):
```bash
python3 scripts/prepare_synapse_classification.py \
  --synapse-root /Users/abdulbasit/Documents/phd_lifetime/endo_agent_project/DATASET_Synapse \
  --task Task002_Synapse \
  --out-root data/synapse_classification_all_organs_multilabel \
  --split-out outputs_synapse_all_organs_multilabel/splits \
  --split 0.8,0.2 \
  --case-split-mode train-val-only \
  --window 50,400 \
  --axis axial \
  --slice-step 1 \
  --class-scheme all-organs \
  --label-policy presence \
  --presence-min-pixels 30 \
  --output-format multilabel \
  --export-images-ts
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

## Normalization (recommended)
Compute dataset mean/std on the train split (saves to `output_dir/normalize.json`):
```bash
python3 scripts/compute_norm.py --config configs/default.yaml
```

## 4-class merge (ERCPMP + Kvasir)
Prepare a merged dataset with videos + images and case-level splits:
```bash
python3 scripts/prepare_ercpmp_kvasir_merge.py \
  --kvasir-zip data/kvasir-dataset-v3.zip \
  --ercpmp-rar data/ERCPMP_v5_Images_Vidoes.rar \
  --out-root data/merged_4class \
  --split-out outputs_4class/splits \
  --fps 1
  --clean
```

Train with the 4-class config:
```bash
python3 src/train.py --config configs/ercpmp_kvasir_4class.yaml
```

Train with Synapse config:
```bash
python3 src/train.py --config configs/synapse.yaml
```

Train with Synapse multilabel config:
```bash
python3 src/train.py --config configs/synapse_multilabel.yaml
```

## Eval
```bash
python3 src/eval.py --config configs/default.yaml --weights outputs/best.pth
```

Multilabel eval:
```bash
python3 src/eval.py --config configs/synapse_multilabel.yaml --weights outputs_synapse_all_organs_multilabel/best.pth
```

## Export
```bash
python3 src/export.py --config configs/default.yaml --weights outputs/best.pth --out outputs/afacnet_cls.pth
```
