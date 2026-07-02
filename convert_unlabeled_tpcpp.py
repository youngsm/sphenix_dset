#!/usr/bin/env python3
"""
Convert the *unlabeled* TPCpp-10M shards (10M events, arXiv:2508.14087) into the
PILArNet-M v2 HDF5 layout consumed by `particle-imaging-models` (pimm), for
self-supervised pretraining (PoLAr-MAE / MAE) where no labels are needed.

SOURCE (TPCpp-10M, Zenodo 10.5281/zenodo.16970029)
---------------------------------------------------
The archive's unlabeled/ dir holds 100 shards:
    unlabeled/spacepoints_000.npz ... _099.npz   (~100k events each -> 10M total)
Each shard, like the labeled spacepoints, has:
    data : (Sum N, 4) float32 = [E, x, y, z]   (x,y,z in cm)
    size : (n_events,) int64  -- points per event; cumsum -> event boundaries
There are NO track_ids / pid_labels / noise_tags for these events.

TARGET (PILArNet-M v2, read by pimm PILArNetH5Dataset.get_data, revision="v2")
------------------------------------------------------------------------------
Same three vlen datasets as the labeled converter, but with placeholder labels:
  point        : vlen float32; reshape(-1,8), reader uses cols [0,1,2,3]=(x,y,z,E).
                 We write [x,y,z,E, E,0,0,0]  (identical to the labeled path).
  cluster      : vlen int32;   reshape(-1,6), reader uses cols [0,2,3,4,5] =
                 (n_pts, group_id, interaction_id, semantic, pid).
                 With no truth, each event is ONE cluster spanning all its points:
                   n_pts=N, group_id=0, interaction_id=0, semantic=1(signal),
                   pid=-1 (UNLABELED / ignore_index). col1 (fragment)=0.
  cluster_extra: vlen float32; reshape(-1,5), reader uses cols [1..4]=(mom,vtx).
                 TPCpp has none -> zeros. col0 set = n_pts (unused).
  <file>_points.npy : (n_events,) int = points per event (pimm build index).

Because there is a single cluster per event, pimm's np.repeat(label, n_pts) yields
group_id=0 / semantic=1 / pid=-1 for every point. Use pid=-1 as the ignore label
for any supervised head; only coord+energy are meaningful for these events.
"""
import argparse
import os
import numpy as np
import h5py

SEM_SIGNAL = 1     # pimm motif "track"
PID_UNLABELED = -1  # ignore_index; no truth PID for unlabeled events


def convert_unlabeled_npz(npz_path, out_h5, gzip=None, verbose=True):
    """Convert one unlabeled spacepoints shard (.npz) -> one pimm-v2 .h5 + _points.npy.

    Writes atomically (out_h5 + '.tmp' then rename) so an interrupted run leaves no
    half-written file to trip up a resume. Returns (n_events, n_points).
    """
    with np.load(npz_path) as h:
        data = h["data"]
        size = h["size"].astype(np.int64)
    assert data.ndim == 2 and data.shape[1] == 4, f"unexpected data shape {data.shape}"
    n_events = len(size)
    offsets = np.concatenate([[0], np.cumsum(size)]).astype(np.int64)
    assert offsets[-1] == len(data), "size cumsum != number of points"

    vlen_f = h5py.vlen_dtype(np.dtype("float32"))
    vlen_i = h5py.vlen_dtype(np.dtype("int32"))
    npoints = np.empty(n_events, dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out_h5)), exist_ok=True)
    tmp_h5 = out_h5 + ".tmp"
    ds_kw = {}
    if gzip is not None:
        ds_kw = dict(compression="gzip", compression_opts=int(gzip))
    with h5py.File(tmp_h5, "w", libver="latest") as f:
        d_point = f.create_dataset("point", shape=(n_events,), dtype=vlen_f, **ds_kw)
        d_clust = f.create_dataset("cluster", shape=(n_events,), dtype=vlen_i, **ds_kw)
        d_extra = f.create_dataset("cluster_extra", shape=(n_events,), dtype=vlen_f, **ds_kw)
        f.attrs["source"] = "TPCpp-10M unlabeled (Zenodo 10.5281/zenodo.16970029, arXiv:2508.14087)"
        f.attrs["revision"] = "v2"
        f.attrs["labeled"] = False
        f.attrs["pid_scheme"] = "unlabeled(pid=-1 ignore)"
        f.attrs["semantic_scheme"] = "1=signal(track); no noise truth for unlabeled"
        f.attrs["shard"] = os.path.basename(npz_path)

        for e in range(n_events):
            s, t = offsets[e], offsets[e + 1]
            E = data[s:t, 0].astype(np.float32)
            xyz = data[s:t, 1:4].astype(np.float32)
            N = xyz.shape[0]
            npoints[e] = N

            # point: [x,y,z,E, E,0,0,0]
            point = np.zeros((N, 8), dtype=np.float32)
            point[:, 0:3] = xyz
            point[:, 3] = E
            point[:, 4] = E
            d_point[e] = point.reshape(-1)

            # cluster (1,6): one cluster spanning the whole event
            # [n_pts, fragment_id, group_id, interaction_id, semantic, pid]
            cluster = np.array([[N, 0, 0, 0, SEM_SIGNAL, PID_UNLABELED]], dtype=np.int32)
            d_clust[e] = cluster.reshape(-1)

            # cluster_extra (1,5): [n_pts, mom, vtx_x, vtx_y, vtx_z] -> zeros
            extra = np.zeros((1, 5), dtype=np.float32)
            extra[0, 0] = N
            d_extra[e] = extra.reshape(-1)

            if verbose and (e % 20000 == 0 or e == n_events - 1):
                print(f"  [{os.path.basename(out_h5)}] {e+1}/{n_events}", flush=True)

    os.replace(tmp_h5, out_h5)
    np.save(out_h5.replace(".h5", "_points.npy"), npoints)
    if verbose:
        print(f"  wrote {out_h5}  ({n_events} events, {int(npoints.sum())} points)")
    return n_events, int(npoints.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="unlabeled spacepoints_*.npz file(s) to convert")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; writes <out>/tpcpp10m_unlabeled_<NNN>.h5")
    ap.add_argument("--gzip", type=int, default=None,
                    help="optional gzip level 1-9 for the HDF5 datasets (smaller, slower)")
    args = ap.parse_args()

    for npz in args.inputs:
        base = os.path.basename(npz)               # spacepoints_000.npz
        idx = base.replace("spacepoints_", "").replace(".npz", "")
        out_h5 = os.path.join(args.out_dir, f"tpcpp10m_unlabeled_{idx}.h5")
        print(f"[{idx}] {npz} -> {out_h5}")
        convert_unlabeled_npz(npz, out_h5, gzip=args.gzip)
    print("done.")


if __name__ == "__main__":
    main()
