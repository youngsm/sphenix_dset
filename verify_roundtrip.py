#!/usr/bin/env python3
"""Validate the converted h5 by replaying pimm's PILArNetH5Dataset.get_data (v2)
logic and comparing against the original TPCpp arrays. No pimm install needed."""
import os, sys, numpy as np, h5py

labeled_dir, h5_path = sys.argv[1], sys.argv[2]

# ---- original TPCpp ----
def ld(n):
    with np.load(os.path.join(labeled_dir, n)) as h: return h["data"], h["size"]
sp, size = ld("spacepoints.npz"); tid,_ = ld("track_ids.npz")
pid,_ = ld("pid_labels.npz"); noise,_ = ld("noise_tags.npz")
off = np.concatenate([[0], np.cumsum(size)]).astype(np.int64)
n_events = len(size)

# ---- replicate pimm get_data(v2) per event ----
def pimm_get_data(f, i):
    data = f["point"][i].reshape(-1, 8)[:, [0,1,2,3]]                 # (x,y,z,e)
    cs, gid, iid, sem, p = f["cluster"][i].reshape(-1, 6)[:, [0,2,-3,-2,-1]].T
    p = p.copy(); p[p == -1] = 5
    ds = np.repeat(sem, cs); dg = np.repeat(gid, cs)
    di = np.repeat(iid, cs); dp = np.repeat(p, cs)
    return data, ds, dg, di, dp

f = h5py.File(h5_path, "r", libver="latest", swmr=True)
pts_npy = np.load(h5_path.replace(".h5", "_points.npy"))

errors = []
# A) global accumulators
tot_pts = 0; pid_hist = np.zeros(6, int); noise_from_sem = 0; noise_from_int = 0
allE_src, allE_h5 = [], []
n_check_coord = 50

for e in range(n_events):
    s, t = off[e], off[e+1]
    src_xyz = sp[s:t, 1:4].astype(np.float32); src_E = sp[s:t, 0].astype(np.float32)
    src_tid = tid[s:t].astype(np.int64); src_pid = pid[s:t].astype(np.int64); src_noise = noise[s:t].astype(bool)

    data, ds, dg, di, dp = pimm_get_data(f, e)
    N = data.shape[0]

    # A) counts
    if N != size[e]:      errors.append(f"ev{e}: N {N} != size {size[e]}")
    if pts_npy[e] != N:   errors.append(f"ev{e}: _points.npy {pts_npy[e]} != {N}")
    tot_pts += N
    for k in range(6): pid_hist[k] += int((dp == k).sum())
    noise_from_sem += int((ds == 4).sum())
    noise_from_int += int((di == -1).sum())
    allE_src.append(src_E); allE_h5.append(data[:,3].astype(np.float32))

    # D) instance count == #unique tracks
    if len(np.unique(dg)) != len(np.unique(src_tid)):
        errors.append(f"ev{e}: n_instances {len(np.unique(dg))} != n_tracks {len(np.unique(src_tid))}")

    # B/C) per-point match by coordinate (first n_check_coord events)
    if e < n_check_coord:
        def key(xyz, E): return {(round(float(a),3),round(float(b),3),round(float(c),3),round(float(g),3))
                                 for a,b,c,g in zip(xyz[:,0],xyz[:,1],xyz[:,2],E)}
        src_keys = key(src_xyz, src_E); h5_keys = key(data[:,:3], data[:,3])
        if src_keys != h5_keys:
            errors.append(f"ev{e}: coord/E multiset mismatch ({len(src_keys)} vs {len(h5_keys)})")
        # match per-point pid+noise via coord->idx
        srcmap = {}
        for j in range(len(src_E)):
            srcmap[(round(float(src_xyz[j,0]),3),round(float(src_xyz[j,1]),3),
                    round(float(src_xyz[j,2]),3),round(float(src_E[j]),3))] = (src_pid[j], bool(src_noise[j]))
        bad = 0
        for j in range(N):
            kk = (round(float(data[j,0]),3),round(float(data[j,1]),3),
                  round(float(data[j,2]),3),round(float(data[j,3]),3))
            if kk in srcmap:
                spid, snoise = srcmap[kk]
                if dp[j] != spid: bad += 1
                if (ds[j]==4) != snoise: bad += 1
                if (di[j]==-1) != snoise: bad += 1
        if bad: errors.append(f"ev{e}: {bad} per-point pid/noise mismatches")

# E) global histograms vs source
src_pid_hist = np.bincount(pid, minlength=6)
if tot_pts != len(sp):                 errors.append(f"total pts {tot_pts} != {len(sp)}")
if not np.array_equal(pid_hist[:5], src_pid_hist[:5]):
    errors.append(f"pid hist {pid_hist[:5].tolist()} != {src_pid_hist[:5].tolist()}")
if noise_from_sem != int(noise.sum()): errors.append(f"noise(sem==4) {noise_from_sem} != {int(noise.sum())}")
if noise_from_int != int(noise.sum()): errors.append(f"noise(int==-1) {noise_from_int} != {int(noise.sum())}")
allE_src = np.sort(np.concatenate(allE_src)); allE_h5 = np.sort(np.concatenate(allE_h5))
if not np.allclose(allE_src, allE_h5):  errors.append("energy values differ")

print(f"events={n_events} total_points={tot_pts}")
print(f"pid_hist(h5)={pid_hist[:5].tolist()}  src={src_pid_hist[:5].tolist()}")
print(f"noise: sem==4 -> {noise_from_sem}, int==-1 -> {noise_from_int}, src={int(noise.sum())}")
print(f"energy match: {np.allclose(allE_src, allE_h5)}")
print(f"coord/E + per-point pid/noise checked on first {n_check_coord} events")
print("RESULT:", "ALL CHECKS PASSED ✅" if not errors else f"{len(errors)} ERRORS ❌")
for e in errors[:20]: print("  -", e)
f.close()
sys.exit(1 if errors else 0)
