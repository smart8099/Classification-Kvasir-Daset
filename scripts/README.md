# Scripts
Add dataset download/prep scripts here.
# sample way to prepare both data

  python3   scripts/prepare_kvasir.py --zip data/kvasir-dataset-v3.zip
  python3 scripts/prepare_ercpmp_kvasir_merge.py \
    --kvasir-zip data/kvasir-dataset-v3.zip \
    --ercpmp-rar data/ERCPMP_v5_Images_Vidoes.rar \
    --out-root data/merged_4class \
    --split-out outputs_4class/splits \
    --fps 1

# Synapse 3D CT segmentation -> 2D classification slices

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

# Train/val-only split and export unlabeled imagesTs slices for inference

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

# True multilabel export for training with BCEWithLogitsLoss

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
