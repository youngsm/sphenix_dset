#!/usr/bin/env python3
"""
Selectively extract members from the TPCpp-10M Zenodo zip WITHOUT downloading
all 118.5 GB. Zenodo serves HTTP range requests (206), so we back Python's
stdlib zipfile with a range-request file object: zipfile reads the (ZIP64)
central directory, then we extract only the chosen members -- pulling just
their bytes (~1 GB for labeled/, vs 118.5 GB for the whole archive).

Usage:
  python fetch_labeled_remote.py --out data/raw --prefix labeled/
  python fetch_labeled_remote.py --out data/raw --prefix labeled/test/ --list
"""
import argparse, io, os, sys, time, random, zipfile, urllib.request, urllib.error, socket

URL = "https://zenodo.org/api/records/16970029/files/TPCpp-10M.zip/content"

# Zenodo rate-limits (HTTP 429) and occasionally 5xx's. Back off + honor Retry-After
# instead of failing, so many parallel range reads eventually all get through.
_RETRY_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 8
_MAX_BACKOFF = 120.0


def _urlopen_retry(req, timeout, read_body=False, max_retries=_MAX_RETRIES):
    """urlopen with exponential backoff on 429/5xx and transient network errors.
    If read_body, returns the response bytes (read fully so a mid-stream reset is
    retried too); otherwise returns the open response for the caller to inspect."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
            if read_body:
                data = r.read()
                r.close()
                return data
            return r
        except urllib.error.HTTPError as e:
            if e.code in _RETRY_CODES and attempt < max_retries - 1:
                ra = e.headers.get("Retry-After") if e.headers else None
                base = float(ra) if (ra and ra.isdigit()) else delay
                # jitter so parallel workers don't wake up and stampede together
                time.sleep(min(base, _MAX_BACKOFF) + random.uniform(0, 3))
                delay = min(delay * 2, _MAX_BACKOFF)
                continue
            raise
        except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError):
            if attempt < max_retries - 1:
                time.sleep(min(delay, _MAX_BACKOFF) + random.uniform(0, 3))
                delay = min(delay * 2, _MAX_BACKOFF)
                continue
            raise
    raise RuntimeError("unreachable")


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file backed by HTTP Range GETs (with backoff)."""
    def __init__(self, url, timeout=60, size=None, probe=True):
        self.url = url; self.timeout = timeout; self._pos = 0
        if size is None:
            req = urllib.request.Request(url, method="HEAD")
            with _urlopen_retry(req, timeout) as r:
                self._size = int(r.headers["Content-Length"])
        else:
            self._size = size
        # verify range support (skippable when the caller already knows the size,
        # e.g. parallel workers that were handed byte-ranges by the parent)
        if probe and not self._probe():
            raise RuntimeError("server does not honor Range requests")

    def _probe(self):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with _urlopen_retry(req, self.timeout) as r:
            return r.status == 206

    # --- io plumbing ---
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self._pos
    def seek(self, off, whence=0):
        if whence == 0: self._pos = off
        elif whence == 1: self._pos += off
        elif whence == 2: self._pos = self._size + off
        return self._pos
    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        if n == 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + n, self._size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self._pos}-{end}"})
        data = _urlopen_retry(req, self.timeout, read_body=True)
        self._pos += len(data)
        return data
    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--prefix", default="labeled/", help="only extract members under this path")
    ap.add_argument("--list", action="store_true", help="just list matching members + sizes")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    rf = HTTPRangeFile(args.url)
    print(f"remote archive: {rf._size/1e9:.2f} GB; range supported.", file=sys.stderr)
    zf = zipfile.ZipFile(rf)
    members = [i for i in zf.infolist()
               if i.filename.startswith(args.prefix) and not i.is_dir()]
    total = sum(i.compress_size for i in members)
    print(f"{len(members)} members under '{args.prefix}', "
          f"~{total/1e9:.3f} GB to transfer:", file=sys.stderr)
    for i in members:
        print(f"  {i.filename}  ({i.file_size/1e6:.1f} MB, stored={i.compress_type==0})")
    if args.list:
        return
    os.makedirs(args.out, exist_ok=True)
    for i in members:
        dest = os.path.join(args.out, i.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"-> {dest}", file=sys.stderr)
        with zf.open(i) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(8 << 20)
                if not chunk: break
                out.write(chunk)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()
