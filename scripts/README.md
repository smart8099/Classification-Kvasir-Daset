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


