#!/usr/bin/env python3
"""
Convert the TPCpp-10M dataset (arXiv:2508.14087, FM4NPP) into the PILArNet-M v2
HDF5 layout consumed by the `particle-imaging-models` (pimm) repo's
`PILArNetH5Dataset` (pimm/datasets/pilarnet.py, revision="v2").

SOURCE (TPCpp-10M, Zenodo 10.5281/zenodo.16970029)
---------------------------------------------------
Each split dir (labeled/{train,val,test}) holds 4 .npz files, each with:
    data : concatenated per-point array for ALL events in the split
    size : (n_events,) int64 -- points per event; cumsum -> event boundaries
  spacepoints.data : (Sum N, 4) float32 = [E, x, y, z]   (x,y,z in cm)
  track_ids.data   : (Sum N,)  uint16   = truth track (instance) id  >= 0
  pid_labels.data  : (Sum N,)  uint8    = {0:other,1:pi+-,2:K+-,3:p,4:e}
  noise_tags.data  : (Sum N,)  bool     = secondary w/ p<60 MeV/c
  (pid and noise are CONSTANT within a track -- verified empirically.)

TARGET (PILArNet-M v2, read by pimm PILArNetH5Dataset.get_data, revision="v2")
------------------------------------------------------------------------------
  point        : vlen float32; per event flat, reshape(-1,8); reader uses cols
                 [0,1,2,3] = (x,y,z,E). We write [x,y,z,E, E,0,0,0].
  cluster      : vlen int32;   per event flat, reshape(-1,6); reader uses cols
                 [0,2,3,4,5] = (n_pts, group_id, interaction_id, semantic, pid).
                 col1 (fragment_id) is unused by pimm; we set it = group_id.
                 MUST be integer dtype: reader does np.repeat(..., n_pts).
  cluster_extra: vlen float32; per event flat, reshape(-1,5); reader uses cols
                 [1,2,3,4] = (momentum, vtx_x, vtx_y, vtx_z). TPCpp has none ->
                 zeros. col0 set = n_pts (unused).
  <file>_points.npy : (n_events,) int = points per event (pimm build index).

  Points within an event are ordered by cluster (sorted by track id) so that
  reader's np.repeat(label, n_pts) reproduces per-point labels.

FULL-FIDELITY label mapping (all 3 TPCpp tasks recoverable in pimm):
  group_id        <- track_id            -> data_dict["instance_particle"]  (track finding)
  segment_pid     <- TPCpp pid (0..4)    -> data_dict["segment_pid"]        (PID)
  semantic_id     <- 1 (signal) / 4 (noise = low-energy-deposit)
                                          -> data_dict["segment_motif"]
  interaction_id  <- 0 (signal) / -1 (noise = background)
                                          -> data_dict["segment_interaction"] (==0 for noise)
  noise mask in pimm == (segment_motif == 4) == (segment_interaction == 0)

  NOTE: pimm's default PID names {0:photon,1:e,2:mu,3:pi,4:p,5:none} DIFFER from
  TPCpp's {0:other,1:pi,2:K,3:p,4:e}. We keep TPCpp's native integers (the user
  chose full-fidelity). Use --remap-pid-to-pimm only if you want pimm's naming.

  NOTE: do NOT rely on pimm's `remove_low_energy_scatters` to drop TPCpp noise --
  that flag drops only cluster[0]. Filter on segment_motif==4 instead.
"""
import argparse
import os
import numpy as np
import h5py

PID_NAMES_TPCPP = {0: "other", 1: "pi+-", 2: "K+-", 3: "proton", 4: "electron"}
# optional remap from TPCpp pid -> pimm pid enum {0:photon,1:e,2:mu,3:pi,4:p,5:none}
# best-effort: pi->3, K-> (no kaon in pimm) -> 5(none), p->4, e->1, other->5
TPCPP_TO_PIMM_PID = {0: 5, 1: 3, 2: 5, 3: 4, 4: 1}

SEM_SIGNAL = 1   # "track"               (pimm motif)
SEM_NOISE = 4    # "low energy deposit"  (pimm motif)


import glob as _glob


def _load_concat(split_dir, stem):
    """Load <stem>.npz, or concatenate sharded <stem>_000.npz, _001.npz, ...
    (train is sharded into 7 parts inside the Zenodo zip)."""
    single = os.path.join(split_dir, f"{stem}.npz")
    if os.path.exists(single):
        files = [single]
    else:
        files = sorted(_glob.glob(os.path.join(split_dir, f"{stem}_*.npz")))
    if not files:
        raise FileNotFoundError(f"no {stem}[.npz|_*.npz] in {split_dir}")
    datas, sizes = [], []
    for fp in files:
        with np.load(fp) as h:
            datas.append(h["data"]); sizes.append(h["size"])
    return np.concatenate(datas, axis=0), np.concatenate(sizes, axis=0)


def load_split(split_dir):
    sp, size = _load_concat(split_dir, "spacepoints")
    tid, _ = _load_concat(split_dir, "track_ids")
    pid, _ = _load_concat(split_dir, "pid_labels")
    noise, _ = _load_concat(split_dir, "noise_tags")
    assert len(tid) == len(pid) == len(noise) == len(sp), "per-point arrays misaligned"
    return sp, size, tid, pid, noise


def convert_split(split_dir, out_h5, remap_pid=False, verbose=True):
    sp, size, tid, pid, noise = load_split(split_dir)
    n_events = len(size)
    offsets = np.concatenate([[0], np.cumsum(size)]).astype(np.int64)
    pid = pid.astype(np.int64)
    if remap_pid:
        lut = np.array([TPCPP_TO_PIMM_PID[i] for i in range(5)], dtype=np.int64)
        pid = lut[pid]

    vlen_f = h5py.vlen_dtype(np.dtype("float32"))
    vlen_i = h5py.vlen_dtype(np.dtype("int32"))
    npoints = np.empty(n_events, dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    with h5py.File(out_h5, "w", libver="latest") as f:
        d_point = f.create_dataset("point", shape=(n_events,), dtype=vlen_f)
        d_clust = f.create_dataset("cluster", shape=(n_events,), dtype=vlen_i)
        d_extra = f.create_dataset("cluster_extra", shape=(n_events,), dtype=vlen_f)
        f.attrs["source"] = "TPCpp-10M (Zenodo 10.5281/zenodo.16970029, arXiv:2508.14087)"
        f.attrs["revision"] = "v2"
        f.attrs["pid_scheme"] = "pimm" if remap_pid else "tpcpp(0:other,1:pi,2:K,3:p,4:e)"
        f.attrs["semantic_scheme"] = "1=signal(track), 4=noise(low-energy-deposit)"

        for e in range(n_events):
            s, t = offsets[e], offsets[e + 1]
            E = sp[s:t, 0].astype(np.float32)
            xyz = sp[s:t, 1:4].astype(np.float32)
            ev_tid = tid[s:t].astype(np.int64)
            ev_pid = pid[s:t]
            ev_noise = noise[s:t]

            # order points by track so clusters are contiguous blocks
            order = np.argsort(ev_tid, kind="stable")
            xyz, E = xyz[order], E[order]
            ev_tid, ev_pid, ev_noise = ev_tid[order], ev_pid[order], ev_noise[order]

            uniq, first_idx, counts = np.unique(
                ev_tid, return_index=True, return_counts=True
            )
            # uniq is sorted asc == block order; first_idx -> first point of each track
            grp = uniq.astype(np.int64)
            cl_noise = ev_noise[first_idx]
            cl_pid = ev_pid[first_idx]
            semantic = np.where(cl_noise, SEM_NOISE, SEM_SIGNAL).astype(np.int64)
            inter = np.where(cl_noise, -1, 0).astype(np.int64)

            N = xyz.shape[0]
            npoints[e] = N

            # point: [x,y,z,E, E,0,0,0]
            point = np.zeros((N, 8), dtype=np.float32)
            point[:, 0:3] = xyz
            point[:, 3] = E
            point[:, 4] = E
            d_point[e] = point.reshape(-1)

            # cluster (-1,6): [n_pts, fragment_id, group_id, interaction_id, semantic, pid]
            cluster = np.empty((len(uniq), 6), dtype=np.int32)
            cluster[:, 0] = counts
            cluster[:, 1] = grp          # fragment_id (unused by pimm) = group_id
            cluster[:, 2] = grp          # group_id -> instance_particle
            cluster[:, 3] = inter        # interaction_id
            cluster[:, 4] = semantic     # semantic / motif
            cluster[:, 5] = cl_pid       # pid
            d_clust[e] = cluster.reshape(-1)

            # cluster_extra (-1,5): [n_pts, mom, vtx_x, vtx_y, vtx_z] -> zeros
            extra = np.zeros((len(uniq), 5), dtype=np.float32)
            extra[:, 0] = counts
            d_extra[e] = extra.reshape(-1)

            if verbose and (e % 5000 == 0 or e == n_events - 1):
                print(f"  [{os.path.basename(out_h5)}] {e+1}/{n_events}", flush=True)

    np.save(out_h5.replace(".h5", "_points.npy"), npoints)
    if verbose:
        print(f"  wrote {out_h5}  ({n_events} events, {npoints.sum()} points) "
              f"+ {os.path.basename(out_h5).replace('.h5','_points.npy')}")
    return n_events, int(npoints.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled-dir", required=True,
                    help="path to extracted 'labeled/' dir (contains train/ val/ test/)")
    ap.add_argument("--out-dir", required=True,
                    help="output root; creates <out>/{train,val,test}/<split>.h5")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="OUTPUT split names (pimm convention: train/val/test)")
    ap.add_argument("--remap-pid-to-pimm", action="store_true",
                    help="remap TPCpp pid -> pimm pid enum (lossy; K+- and other -> 'none')")
    args = ap.parse_args()

    # output split name -> candidate source dir names (TPCpp uses 'validation')
    SRC_DIRS = {"train": ["train"], "val": ["val", "validation"],
                "test": ["test"], "validation": ["validation"]}

    for split in args.splits:
        src = None
        for cand in SRC_DIRS.get(split, [split]):
            p = os.path.join(args.labeled_dir, cand)
            if os.path.isdir(p):
                src = p; break
        if src is None:
            print(f"[skip] no source dir for split '{split}' under {args.labeled_dir}")
            continue
        out_split = "val" if split == "validation" else split
        out_h5 = os.path.join(args.out_dir, out_split, f"tpcpp10m_{out_split}.h5")
        print(f"[{out_split}] {src} -> {out_h5}")
        convert_split(src, out_h5, remap_pid=args.remap_pid_to_pimm)
    print("done.")


if __name__ == "__main__":
    main()
