#!/usr/bin/env python3
"""
Stream the 10M *unlabeled* TPCpp events from the Zenodo archive and convert them
to PILArNet-M v2 HDF5 -- in parallel, one shard at a time, WITHOUT ever holding
the whole 117 GB on disk.

For each of the 100 shards (unlabeled/spacepoints_NNN.npz) a worker:
  1. range-extracts just that member from the remote zip (~1.17 GB),
  2. converts it to <h5-dir>/tpcpp10m_unlabeled_NNN.h5 (+ _points.npy),
  3. deletes the staged .npz  (unless --keep-raw).

Fully resumable: shards whose .h5 already exists are skipped. Peak disk is about
   jobs * (1.17 GB npz + its .h5)   and peak RAM about   jobs * ~4 GB
so pick --jobs to fit the node. Downloads dominate wall time, so it is fine to set
--jobs a bit higher than cores.

Usage (typical, on the cluster):
  python fetch_convert_unlabeled.py --out /scratch/$USER/tpcpp --jobs 16
  python fetch_convert_unlabeled.py --out /scratch/$USER/tpcpp --shards 0-9   # a subset
"""
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import zipfile

from fetch_labeled_remote import HTTPRangeFile, URL
from convert_unlabeled_tpcpp import convert_unlabeled_npz

N_SHARDS = 100
MEMBER = "unlabeled/spacepoints_{:03d}.npz"


def parse_shards(spec):
    """'0-9', '0,3,5-7', '42' -> sorted list of unique ints (clamped to 0..99)."""
    if spec is None:
        return list(range(N_SHARDS))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [s for s in out if not 0 <= s < N_SHARDS]
    if bad:
        raise SystemExit(f"shard(s) out of range 0..{N_SHARDS-1}: {sorted(bad)}")
    return sorted(out)


def _download_member(url, member, dest, chunk=8 << 20):
    """Range-extract one zip member to dest (atomic via .tmp)."""
    rf = HTTPRangeFile(url)
    zf = zipfile.ZipFile(rf)
    info = zf.getinfo(member)
    tmp = dest + ".tmp"
    with zf.open(info) as src, open(tmp, "wb") as out:
        while True:
            b = src.read(chunk)
            if not b:
                break
            out.write(b)
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def process_shard(idx, url, raw_dir, h5_dir, gzip, keep_raw):
    tag = f"{idx:03d}"
    out_h5 = os.path.join(h5_dir, f"tpcpp10m_unlabeled_{tag}.h5")
    pts_npy = out_h5.replace(".h5", "_points.npy")
    if os.path.exists(out_h5) and os.path.exists(pts_npy):
        return (idx, "skip", 0, 0)
    member = MEMBER.format(idx)
    npz_path = os.path.join(raw_dir, f"spacepoints_{tag}.npz")
    try:
        if not os.path.exists(npz_path):
            print(f"[{tag}] downloading {member} ...", flush=True)
            nbytes = _download_member(url, member, npz_path)
            print(f"[{tag}] downloaded {nbytes/1e9:.2f} GB; converting ...", flush=True)
        else:
            print(f"[{tag}] found staged npz; converting ...", flush=True)
        n_ev, n_pt = convert_unlabeled_npz(npz_path, out_h5, gzip=gzip, verbose=False)
        if not keep_raw:
            os.remove(npz_path)
        print(f"[{tag}] done: {n_ev} events, {n_pt} points -> {out_h5}", flush=True)
        return (idx, "ok", n_ev, n_pt)
    except Exception as e:
        traceback.print_exc()
        return (idx, f"ERROR: {e}", 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                    help="parallel shards in flight (default min(8, ncpu))")
    ap.add_argument("--shards", default=None,
                    help="which shards, e.g. '0-99' (default), '0-9', '0,3,5-7'")
    ap.add_argument("--raw-dir", default=None, help="stage dir for .npz (default <out>/raw_unlabeled)")
    ap.add_argument("--h5-dir", default=None, help="output dir for .h5 (default <out>/pilarnet_v2/unlabeled)")
    ap.add_argument("--gzip", type=int, default=None, help="gzip level 1-9 for HDF5 (smaller, slower)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the staged .npz after converting")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out, "raw_unlabeled")
    h5_dir = args.h5_dir or os.path.join(args.out, "pilarnet_v2", "unlabeled")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(h5_dir, exist_ok=True)

    shards = parse_shards(args.shards)
    print(f">> {len(shards)} shard(s), jobs={args.jobs}", file=sys.stderr)
    print(f">> raw stage: {raw_dir}", file=sys.stderr)
    print(f">> h5 out:    {h5_dir}", file=sys.stderr)

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_shard, i, args.url, raw_dir, h5_dir,
                          args.gzip, args.keep_raw): i for i in shards}
        for fut in as_completed(futs):
            results.append(fut.result())

    ok = [r for r in results if r[1] == "ok"]
    skip = [r for r in results if r[1] == "skip"]
    err = [r for r in results if r[1].startswith("ERROR")]
    tot_ev = sum(r[2] for r in results)
    tot_pt = sum(r[3] for r in results)
    print(f"\n>> converted {len(ok)} shard(s), skipped {len(skip)}, "
          f"{len(err)} error(s); {tot_ev} events, {tot_pt} points.", file=sys.stderr)
    if err:
        for idx, msg, _, _ in sorted(err):
            print(f"   shard {idx:03d}: {msg}", file=sys.stderr)
        print(">> re-run the SAME command to retry only the failed/missing shards.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
