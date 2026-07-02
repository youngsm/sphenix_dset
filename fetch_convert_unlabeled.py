#!/usr/bin/env python3
"""
Stream the 10M *unlabeled* TPCpp events from the Zenodo archive and convert them
to PILArNet-M v2 HDF5 -- in parallel, one shard at a time, WITHOUT ever holding
the whole 117 GB on disk.

The parent reads the zip's central directory ONCE and resolves each shard's exact
compressed byte-range. Each worker then:
  1. pulls just that byte-range in a few large raw Range GETs and inflates it
     locally (no per-worker central-directory reads -> ~10-20 requests/shard),
  2. converts the .npz to <h5-dir>/tpcpp10m_unlabeled_NNN.h5 (+ _points.npy),
  3. deletes the staged .npz  (unless --keep-raw).

Zenodo rate-limits (HTTP 429); HTTPRangeFile backs off and honors Retry-After, and
big chunks keep the request count low, so the run self-paces instead of failing.
If you still see frequent "429" pauses, lower --jobs or raise --chunk-mb.

Fully resumable: shards whose .h5 already exists are skipped. Peak disk is about
   jobs * (1.17 GB npz + its .h5)   and peak RAM about   jobs * ~4 GB.

Usage (typical, on the cluster):
  python fetch_convert_unlabeled.py --out /scratch/$USER/tpcpp --jobs 4
  python fetch_convert_unlabeled.py --out /scratch/$USER/tpcpp --shards 0-9   # a subset
"""
import argparse
import os
import socket
import struct
import sys
import time
import traceback
import urllib.error
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import zipfile

from fetch_labeled_remote import HTTPRangeFile, URL
from convert_unlabeled_tpcpp import convert_unlabeled_npz

N_SHARDS = 100
MEMBER = "unlabeled/spacepoints_{:03d}.npz"

# Exceptions that mean "can't reach the host" (DNS / connect / proxy), as opposed
# to a bug in our code. We surface these once, cleanly, instead of per-worker.
NET_ERRORS = (urllib.error.URLError, socket.gaierror, socket.timeout, ConnectionError, TimeoutError)


def _is_net_error(exc):
    return isinstance(exc, NET_ERRORS) or isinstance(getattr(exc, "reason", None), NET_ERRORS)


def parse_shards(spec):
    """'0-9', '0,3,5-7', '42' -> sorted list of unique ints (validated to 0..99)."""
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


def _net_help(url, err):
    host = url.split("/")[2]
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    return "\n".join([
        f"cannot reach '{host}' from this node ({socket.gethostname()}): {err}",
        "",
        "This is a network problem, not a data problem. Likely one of:",
        "  1. This node has no outbound internet. Run from a node that does",
        "     (a login or data-transfer node), or submit the job there.",
        "  2. Your site needs an HTTP proxy. urllib honors https_proxy/http_proxy:",
        "       export https_proxy=http://<proxy-host>:<port>",
        "       export http_proxy=http://<proxy-host>:<port>",
        "     (SLAC S3DF nodes typically need a squid proxy; check the S3DF docs.)",
        f"       currently https_proxy={proxy!r}",
        "",
        "Verify with:  python3 -c \"import urllib.request as u; "
        f"print(u.urlopen(u.Request('{url}', method='HEAD'), timeout=30).status)\"",
    ])


def resolve_members(url, shards):
    """Read the central directory ONCE and return (archive_size, {idx: spec}) where
    spec = (data_off, compress_size, compress_type, crc32, uncompressed_size).
    Also serves as a fast-failing connectivity preflight."""
    rf = HTTPRangeFile(url)
    zf = zipfile.ZipFile(rf)
    specs = {}
    for idx in shards:
        info = zf.getinfo(MEMBER.format(idx))
        rf.seek(info.header_offset)
        lh = rf.read(30)
        if lh[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local file header for shard {idx:03d}")
        fnlen, exlen = struct.unpack("<HH", lh[26:30])
        data_off = info.header_offset + 30 + fnlen + exlen
        specs[idx] = (data_off, info.compress_size, info.compress_type, info.CRC, info.file_size)
    return rf._size, specs


def _download_inflate(url, archive_size, spec, dest, chunk):
    """Pull one member's compressed byte-range and inflate to dest (atomic .tmp),
    verifying size + CRC32. Uses a fixed archive size + no Range probe so a worker
    spends its requests on data, not metadata."""
    data_off, csize, ctype, crc, usize = spec
    rf = HTTPRangeFile(url, size=archive_size, probe=False)
    rf.seek(data_off)
    dec = zlib.decompressobj(-15) if ctype == zipfile.ZIP_DEFLATED else None
    if ctype not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise RuntimeError(f"unsupported compress_type {ctype}")
    run_crc, n_out, remaining = 0, 0, csize
    tmp = dest + ".tmp"
    with open(tmp, "wb") as out:
        while remaining > 0:
            block = rf.read(min(chunk, remaining))
            if not block:
                raise IOError(f"short read: {remaining} compressed bytes left")
            remaining -= len(block)
            piece = dec.decompress(block) if dec else block
            if piece:
                out.write(piece); run_crc = zlib.crc32(piece, run_crc); n_out += len(piece)
        if dec:
            piece = dec.flush()
            if piece:
                out.write(piece); run_crc = zlib.crc32(piece, run_crc); n_out += len(piece)
    if n_out != usize:
        raise IOError(f"decompressed size {n_out} != expected {usize}")
    if crc and (run_crc & 0xffffffff) != (crc & 0xffffffff):
        raise IOError("CRC32 mismatch (corrupt download)")
    os.replace(tmp, dest)
    return n_out


def process_shard(idx, url, archive_size, spec, raw_dir, h5_dir, gzip, keep_raw, chunk, debug=False):
    tag = f"{idx:03d}"
    out_h5 = os.path.join(h5_dir, f"tpcpp10m_unlabeled_{tag}.h5")
    pts_npy = out_h5.replace(".h5", "_points.npy")
    if os.path.exists(out_h5) and os.path.exists(pts_npy):
        return (idx, "skip", 0, 0)
    npz_path = os.path.join(raw_dir, f"spacepoints_{tag}.npz")
    try:
        if not os.path.exists(npz_path):
            print(f"[{tag}] downloading {MEMBER.format(idx)} "
                  f"({spec[1]/1e9:.2f} GB compressed) ...", flush=True)
            _download_inflate(url, archive_size, spec, npz_path, chunk)
            print(f"[{tag}] downloaded; converting ...", flush=True)
        else:
            print(f"[{tag}] found staged npz; converting ...", flush=True)
        n_ev, n_pt = convert_unlabeled_npz(npz_path, out_h5, gzip=gzip, verbose=False)
        if not keep_raw:
            os.remove(npz_path)
        print(f"[{tag}] done: {n_ev} events, {n_pt} points -> {out_h5}", flush=True)
        return (idx, "ok", n_ev, n_pt)
    except Exception as e:  # noqa: BLE001
        # leave no half-written files for the resume to trip on
        for junk in (npz_path, npz_path + ".tmp", out_h5 + ".tmp"):
            try:
                os.remove(junk)
            except OSError:
                pass
        kind = "NET" if _is_net_error(e) else "ERROR"
        if debug:
            traceback.print_exc()
        print(f"[{tag}] {kind}: {e}", flush=True)
        return (idx, f"{kind}: {e}", 0, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel shards in flight (default 4; lower if you hit 429s)")
    ap.add_argument("--shards", default=None,
                    help="which shards, e.g. '0-99' (default), '0-9', '0,3,5-7'")
    ap.add_argument("--chunk-mb", type=int, default=128,
                    help="Range-GET size in MB (default 128; bigger = fewer requests, more RAM)")
    ap.add_argument("--raw-dir", default=None, help="stage dir for .npz (default <out>/raw_unlabeled)")
    ap.add_argument("--h5-dir", default=None, help="output dir for .h5 (default <out>/pilarnet_v2/unlabeled)")
    ap.add_argument("--gzip", type=int, default=None, help="gzip level 1-9 for HDF5 (smaller, slower)")
    ap.add_argument("--keep-raw", action="store_true", help="keep the staged .npz after converting")
    ap.add_argument("--debug", action="store_true", help="print full tracebacks on per-shard errors")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out, "raw_unlabeled")
    h5_dir = args.h5_dir or os.path.join(args.out, "pilarnet_v2", "unlabeled")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(h5_dir, exist_ok=True)
    chunk = args.chunk_mb << 20

    shards = parse_shards(args.shards)
    todo = [i for i in shards
            if not (os.path.exists(os.path.join(h5_dir, f"tpcpp10m_unlabeled_{i:03d}.h5"))
                    and os.path.exists(os.path.join(h5_dir, f"tpcpp10m_unlabeled_{i:03d}_points.npy")))]
    print(f">> {len(shards)} shard(s) requested, {len(todo)} to do "
          f"({len(shards)-len(todo)} already done); jobs={args.jobs}, chunk={args.chunk_mb} MB",
          file=sys.stderr)
    print(f">> raw stage: {raw_dir}", file=sys.stderr)
    print(f">> h5 out:    {h5_dir}", file=sys.stderr)
    if not todo:
        print(">> nothing to do.", file=sys.stderr)
        return

    print(">> resolving member byte-ranges (reading central directory once) ...", file=sys.stderr)
    try:
        archive_size, specs = resolve_members(args.url, todo)
    except Exception as e:  # noqa: BLE001
        if _is_net_error(e):
            print("\n" + _net_help(args.url, e), file=sys.stderr)
            sys.exit(2)
        raise

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_shard, i, args.url, archive_size, specs[i],
                          raw_dir, h5_dir, args.gzip, args.keep_raw, chunk, args.debug): i
                for i in todo}
        for fut in as_completed(futs):
            results.append(fut.result())

    ok = [r for r in results if r[1] == "ok"]
    err = [r for r in results if r[1].startswith(("ERROR", "NET"))]
    tot_ev = sum(r[2] for r in results)
    tot_pt = sum(r[3] for r in results)
    print(f"\n>> converted {len(ok)} shard(s), {len(err)} error(s); "
          f"{tot_ev} events, {tot_pt} points.", file=sys.stderr)
    if err:
        for idx, msg, _, _ in sorted(err):
            print(f"   shard {idx:03d}: {msg}", file=sys.stderr)
        print(">> re-run the SAME command to retry only the failed/missing shards.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
