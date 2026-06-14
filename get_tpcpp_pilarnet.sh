#!/usr/bin/env bash
# Download TPCpp-10M (arXiv:2508.14087) labeled splits and convert them to the
# PILArNet-M v2 HDF5 layout used by `particle-imaging-models` (pimm).
#
# Run this ON THE HPC where you want the data. Two modes:
#   (default) labeled-only via remote partial-zip extraction  -> ~1.05 GB transfer
#   --full    download the whole 118.5 GB archive (resumable) -> then extract labeled/
#
# Usage:
#   ./get_tpcpp_pilarnet.sh /path/to/output_root            # labeled-only (recommended)
#   ./get_tpcpp_pilarnet.sh /path/to/output_root --full     # full archive
set -euo pipefail

OUT="${1:?usage: $0 <output_root> [--full]}"
MODE="${2:-labeled}"
HERE="$(cd "$(dirname "$0")" && pwd)"

RAW="$OUT/raw"
H5="$OUT/pilarnet_v2"
ZURL="https://zenodo.org/api/records/16970029/files"
MD5_MAIN="04f21ccf0aa0251ec40130f9c61223f7"

mkdir -p "$RAW" "$H5"
echo ">> output root: $OUT"
python3 -c "import numpy, h5py" || { echo "ERROR: need numpy + h5py (pip install numpy h5py)"; exit 1; }

if [ "$MODE" = "--full" ]; then
  echo ">> FULL mode: downloading TPCpp-10M.zip (118.5 GB, resumable) ..."
  wget -c -O "$RAW/TPCpp-10M.zip" "$ZURL/TPCpp-10M.zip/content"
  echo ">> verifying md5 ..."
  if command -v md5sum >/dev/null; then GOT=$(md5sum "$RAW/TPCpp-10M.zip" | awk '{print $1}');
  else GOT=$(md5 -q "$RAW/TPCpp-10M.zip"); fi
  [ "$GOT" = "$MD5_MAIN" ] || { echo "MD5 MISMATCH ($GOT != $MD5_MAIN)"; exit 1; }
  echo ">> extracting only labeled/ (skips ~117 GB of unlabeled/) ..."
  unzip -o "$RAW/TPCpp-10M.zip" "labeled/*" -d "$RAW"
else
  echo ">> LABELED-ONLY mode: partial-extracting labeled/ from remote zip (~1.05 GB) ..."
  python3 "$HERE/fetch_labeled_remote.py" --prefix labeled/ --out "$RAW"
fi

echo ">> converting to PILArNet-M v2 HDF5 ..."
python3 "$HERE/convert_tpcpp_to_pilarnet.py" \
  --labeled-dir "$RAW/labeled" --out-dir "$H5" --splits train val test

cat <<EOF

>> DONE.
   Converted HDF5 (pimm v2) is at: $H5/{train,val,test}/*.h5  (+ *_points.npy)

   Point pimm at it via PILARNET_DATA_ROOT_V2=$H5  (or data_root: $H5),
   split names train/val/test, revision="v2".
   The converter already wrote *_points.npy, so pimm's build_index is optional:
     python -m polarmae.datasets.build_index "$H5"/**/*.h5 -j 8   # if you prefer
EOF
