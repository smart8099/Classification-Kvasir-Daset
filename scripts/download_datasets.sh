#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-"$PWD/.data"}
mkdir -p "$DATA_ROOT"

echo "Public dataset sources (per EndoAgent paper):"

echo "Kvasir (classification images):"
echo "  - https://datasets.simula.no/kvasir/"

echo "Kvasir-SEG (segmentation masks):"
echo "  - https://datasets.simula.no/kvasir-seg/"

echo "CVC-ClinicDB (GIANA Grand Challenge):"
echo "  - https://polyp.grand-challenge.org/CVCClinicDB/"

echo "ETIS-Larib Polyp DB (Grand Challenge):"
echo "  - https://polyp.grand-challenge.org/ETISLarib/"

echo "CVC-ColonDB (CVC UAB):"
echo "  - http://vi.cvc.uab.es/colon-qa/cvccolondb/"

echo "CVC-300 / EndoScene-CVC300 (CVC UAB):"
echo "  - https://pages.cvc.uab.es/CVC-Colon/"

echo "SUN-SEG (VPS / SUN-SEG dataset):"
echo "  - https://github.com/GewelsJI/VPS (see DATA_PREPARATION; access requires email authorization)"

echo

echo "Notes:"
echo "- Some datasets require registration/authorization (Grand Challenge, SUN-SEG)."
echo "- Please review each dataset's license/terms before use."

echo

echo "This script only prints sources. For automated downloads,"
echo "add curl/wget commands after you obtain access links or tokens."
