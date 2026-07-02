#!/usr/bin/env bash
# Download the 10M UNLABELED TPCpp events (arXiv:2508.14087) and convert them to
# the PILArNet-M v2 HDF5 layout used by `particle-imaging-models` (pimm), for
# self-supervised pretraining. Same format as the labeled data, just no labels
# (pid=-1, single cluster per event).
#
# Streams shard-by-shard in parallel: downloads ~1.17 GB, converts, deletes the
# raw npz, moves on -- so peak disk stays ~ jobs * a couple GB, not 117 GB.
# Fully resumable: re-run the same command to pick up any missing/failed shards.
#
# Run this ON THE CLUSTER where you want the data.
#
# Usage:
#   ./get_tpcpp_unlabeled.sh <output_root> [jobs] [shards]
#
# Examples:
#   ./get_tpcpp_unlabeled.sh /scratch/$USER/tpcpp            # all 100 shards, 8 jobs
#   ./get_tpcpp_unlabeled.sh /scratch/$USER/tpcpp 16         # all 100 shards, 16 jobs
#   ./get_tpcpp_unlabeled.sh /scratch/$USER/tpcpp 16 0-9     # just shards 0-9
set -euo pipefail

OUT="${1:?usage: $0 <output_root> [jobs] [shards]}"
JOBS="${2:-8}"
SHARDS="${3:-0-99}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo ">> output root: $OUT   jobs: $JOBS   shards: $SHARDS"
python3 -c "import numpy, h5py" || { echo "ERROR: need numpy + h5py (pip install numpy h5py)"; exit 1; }

python3 "$HERE/fetch_convert_unlabeled.py" \
  --out "$OUT" --jobs "$JOBS" --shards "$SHARDS"

cat <<EOF

>> DONE.
   Unlabeled HDF5 (pimm v2) is at: $OUT/pilarnet_v2/unlabeled/tpcpp10m_unlabeled_*.h5  (+ *_points.npy)

   Use it as a pretraining split: point pimm at PILARNET_DATA_ROOT_V2=$OUT/pilarnet_v2
   with split "unlabeled", revision="v2". Only coord+energy are meaningful; pid=-1
   is an ignore label (single cluster per event, no truth track/pid/noise).
EOF
