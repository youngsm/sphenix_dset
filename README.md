# TPCpp-10M → PILArNet-M (pimm) converter

Download the dataset from **FM4NPP: A Scaling Foundation Model for Nuclear and
Particle Physics** ([arXiv:2508.14087](https://arxiv.org/abs/2508.14087)) and
convert it into the **PILArNet-M v2** HDF5 layout consumed by
[`particle-imaging-models`](https://github.com/youngsm/particle-imaging-models)
(`pimm`), so you can fine-tune / evaluate on the TPCpp tasks.

## Source dataset

- **Name:** TPCpp-10M — simulated p+p collisions @ √s=200 GeV in the sPHENIX TPC
  (PYTHIA 8.307 + GEANT4). Data descriptor: *Data in Brief* `10.1016/j.dib.2025.112393`.
- **Host:** Zenodo **DOI [10.5281/zenodo.16970029](https://doi.org/10.5281/zenodo.16970029)**, license **CC BY 4.0**.
- **Files:** `TPCpp-10M.zip` (118.5 GB: 10M unlabeled + labeled train/val),
  `TPCpp-10M_labeled_test.zip` (82 MB), `TPCpp-10M_scripts.zip` (demo).
- **Labeled splits:** train 70k (7 shards), validation 13k, test 7k events.
  Each split has `spacepoints.npz [E,x,y,z]`, `track_ids.npz` (instance),
  `pid_labels.npz` {0:other,1:π±,2:K±,3:p,4:e}, `noise_tags.npz` (bool, p<60 MeV/c
  secondaries). `pid` and `noise` are constant within a truth track (verified).

> The labeled **train/val live inside the 118.5 GB zip**; only the test set is a
> separate file. Zenodo supports HTTP range requests, so we extract just the
> `labeled/` members (~1.05 GB) without downloading the full archive.

## Quick start (run on the HPC)

```bash
pip install numpy h5py            # only deps
./get_tpcpp_pilarnet.sh /scratch/$USER/tpcpp            # labeled-only, ~1.05 GB transfer
# or, if you also want the 10M unlabeled later:
./get_tpcpp_pilarnet.sh /scratch/$USER/tpcpp --full     # 118.5 GB, resumable
```

Output: `…/pilarnet_v2/{train,val,test}/tpcpp10m_<split>.h5` + `*_points.npy`.
Point pimm at it with `PILARNET_DATA_ROOT_V2=…/pilarnet_v2`, splits `train/val/test`,
`revision="v2"`.

### Pieces (if you want to run them individually)
```bash
python3 fetch_labeled_remote.py --prefix labeled/ --out RAW          # partial-zip pull
python3 convert_tpcpp_to_pilarnet.py --labeled-dir RAW/labeled --out-dir OUT_H5
python3 verify_roundtrip.py RAW/labeled/test OUT_H5/test/tpcpp10m_test.h5   # optional check
```

## Format mapping (TPCpp → pimm v2)

`PILArNetH5Dataset.get_data(revision="v2")` reads:
`point.reshape(-1,8)[:, :4]` = (x,y,z,E); `cluster.reshape(-1,6)[:, [0,2,3,4,5]]`
= (n_pts, group_id, interaction_id, semantic, pid); `cluster_extra.reshape(-1,5)[:, 1:]`
= (momentum, vtx). Points are stored ordered by cluster.

| pimm field (`data_dict`) | source of value | TPCpp task |
|---|---|---|
| `coord` (x,y,z) | spacepoints x,y,z (cm) | — |
| `energy` | spacepoints E (point col 3) | — |
| `instance_particle` | `group_id` ← `track_id` | **track finding** |
| `segment_pid` | TPCpp `pid` (raw 0..4) | **particle ID** |
| `segment_motif` | 1=signal / **4**=noise | **noise tagging** |
| `segment_interaction` | 0 for noise / 1 for signal (via `interaction_id` −1/0) | noise tagging |
| `momentum`,`vertex` | 0 (not in TPCpp) | — |

Noise is recoverable as `segment_motif==4` **or** `segment_interaction==0`.

## Caveats

- **PID enum differs.** We keep TPCpp's native ints `{0:other,1:π,2:K,3:p,4:e}`,
  which are **not** pimm's default `{0:γ,1:e,2:μ,3:π,4:p,5:none}`. Set your PID head
  to 5 classes with the TPCpp meaning, or pass `--remap-pid-to-pimm` (lossy: K± and
  "other" → "none").
- **Energy scale.** TPCpp E ∈ ~[23, 6250] (ionization, arbitrary units), not LArTPC
  MeV. Re-tune `energy_threshold` / `emin` / `emax`; the PoLAr-MAE defaults
  (0.13 / 1e-2 / 20) assume MeV and will be wrong here.
- **Coordinates** are centered at 0 (x,y∈±75, z∈±105 cm). pimm's event-overlay
  rotation assumes a center of (384,384,384) — leave `overlay_n_events=1` (default).
- **`remove_low_energy_scatters`** drops only `cluster[0]`; with TPCpp, noise is many
  clusters, so don't use it to remove noise — filter `segment_motif==4` instead.

## Validation performed

Converted all 7,000 real test events and replayed pimm's exact v2 reader; against the
source: total points, PID histogram, energy values, and noise (both encodings) match
exactly; per-point coord/PID/noise verified on 50 events; instance count == #tracks.
Remote partial extraction verified byte-identical to the standalone test archive.
