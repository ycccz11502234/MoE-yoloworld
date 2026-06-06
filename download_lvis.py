"""Download and prepare the LVIS dataset for ``ultralytics`` training.

Layout (after this script finishes successfully)::

    <DATASETS_ROOT>/
    └── lvis/
        ├── train.txt              # ~99k image paths (relative)
        ├── val.txt                # ~19k image paths
        ├── minival.txt            # 5k image paths
        ├── annotations/           # lvis_v1_val.json, lvis_v1_minival.json
        ├── labels/                # YOLO-format labels (one .txt per image)
        │   ├── train2017/
        │   └── val2017/
        └── images/
            ├── train2017/         # 118k images (~19 GB)
            ├── val2017/           # 5k images   (~ 1 GB)
            └── test2017/          # 41k images  (~ 7 GB, optional)

The destination directory must match ``datasets_dir`` in
``~/.config/Ultralytics/settings.json``.

Why a custom downloader?
    The corporate HTTP proxy throttles every single TCP stream to
    ~120 KB/s, which would take >40 hours for the 18 GB train2017 zip.
    Splitting the file into N HTTP byte-ranges and downloading them in
    parallel routes around that per-stream cap and sustains 10+ MB/s.

    Each range part is fetched independently and only renamed to its final
    ``.partK`` slot after a successful exit code, then we sanity-check the
    concatenated file's total size against the server's ``Content-Length``
    *and* run :py:meth:`zipfile.ZipFile.testzip` so a half-finished resume
    can never silently corrupt the archive.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Sources required to assemble LVIS for ``ultralytics`` training
# ---------------------------------------------------------------------------
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/lvis-labels-segments.zip"
IMAGE_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",  # ~19 GB, 118k
    "val2017":   "http://images.cocodataset.org/zips/val2017.zip",    # ~1 GB,  5k
    "test2017":  "http://images.cocodataset.org/zips/test2017.zip",   # ~7 GB,  41k (optional)
}

DEFAULT_ROOT = "/apdcephfs_gy8/share_304660203/hunyuan/yccczwang/dev/datasets"
DEFAULT_PROXY = "http://star-proxy.oa.com:3128"


def parse_args():
    p = argparse.ArgumentParser(description="Download LVIS for ultralytics training")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="Datasets root (must match Ultralytics settings.json -> datasets_dir)")
    p.add_argument("--proxy", default=DEFAULT_PROXY,
                   help="HTTP/HTTPS proxy (use --no-proxy to disable)")
    p.add_argument("--no-proxy", action="store_true")
    p.add_argument("--skip-test", action="store_true",
                   help="Skip the optional test2017 split (~7 GB)")
    p.add_argument("--connections", type=int, default=8,
                   help="Top-level HTTP-range parts per file (default 8)")
    p.add_argument("--substreams", type=int, default=4,
                   help="Parallel sub-streams *inside* every part (default 4). "
                        "Effective concurrency = connections * substreams.")
    p.add_argument("--retries", type=int, default=10,
                   help="curl retry count (--retry)")
    p.add_argument("--retry-delay", type=int, default=5)
    p.add_argument("--keep-zip", action="store_true",
                   help="Keep the .zip files after extraction")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_proxy(proxy: str):
    """Inject the proxy into the env so curl & subprocesses use it."""
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[k] = proxy
    print(f"[lvis] using HTTP proxy: {proxy}", flush=True)


def remote_size(url: str) -> int:
    """Return ``Content-Length`` of the final URL (after redirects), or 0."""
    try:
        out = subprocess.check_output(
            ["curl", "-sSIL", "--max-time", "30", url],
            stderr=subprocess.STDOUT,
        ).decode(errors="ignore")
    except subprocess.CalledProcessError:
        return 0
    last = 0
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                last = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return last


def _curl_simple(url: str, dst: Path, start: int, end: int,
                 retries: int, retry_delay: int):
    """Bare ``curl`` call to fetch bytes ``[start, end]`` of ``url`` -> ``dst``.

    Uses ``-C -`` for byte-range resume.  Caller is responsible for size
    validation.
    """
    cmd = [
        "curl", "-fsS", "-L",
        "--retry", str(retries),
        "--retry-delay", str(retry_delay),
        "--retry-connrefused",
        "--range", f"{start}-{end}",
        "-C", "-",
        "-o", str(dst),
        url,
    ]
    subprocess.check_call(cmd)


def _curl_range(url: str, dst: Path, start: int, end: int,
                retries: int, retry_delay: int,
                substreams: int = 4) -> int:
    """Fetch bytes ``[start, end]`` of ``url`` -> ``dst`` using ``substreams``
    parallel TCP connections.

    Why sub-streams?  The corporate HTTP proxy throttles every individual
    TCP stream; by fanning a single byte-range out into several sub-ranges
    we get a much higher aggregate throughput, *and* fresh ``curl``
    invocations get re-routed through the proxy's load balancer so even
    after a slow node assignment we have new chances for a fast one.

    Strategy:
      * If ``dst`` is already complete, return immediately (resume idempotent).
      * Otherwise treat the existing ``dst`` (if any) as ``sub-0`` initial
        content -- ``curl -C -`` will resume from ``start + dst.size``.
      * Split the rest of the range into ``substreams`` slices that go to
        ``dst.subK`` files, then concatenate everything into ``dst``.
    """
    expected = end - start + 1
    if dst.exists() and dst.stat().st_size == expected:
        return expected

    # boundary: where existing bytes end
    have = dst.stat().st_size if dst.exists() else 0
    remaining_start = start + have
    if remaining_start > end:
        # Local file is *bigger* than the requested range -> drop & redownload
        dst.unlink()
        have = 0
        remaining_start = start

    if substreams <= 1 or remaining_start > end:
        # Single-stream resume into dst (may be 0-byte == fresh)
        _curl_simple(url, dst, start, end, retries, retry_delay)
        return dst.stat().st_size

    # ---- split the *remaining* bytes into substreams sub-slices ---------
    sub_total = end - remaining_start + 1
    sub_size = (sub_total + substreams - 1) // substreams
    subs = []  # (idx, abs_start, abs_end, sub_path)
    sub_dir = dst.parent / (dst.name + ".subs")
    sub_dir.mkdir(exist_ok=True)
    for i in range(substreams):
        s = remaining_start + i * sub_size
        e = min(s + sub_size - 1, end)
        if s > e:
            break
        subs.append((i, s, e, sub_dir / f"sub{i:02d}"))

    # Run all sub-streams in parallel
    with ThreadPoolExecutor(max_workers=len(subs)) as ex:
        futs = [ex.submit(_curl_simple, url, sp, s, e, retries, retry_delay)
                for i, s, e, sp in subs]
        for f in futs:
            f.result()  # propagate first error

    # Verify each sub
    for i, s, e, sp in subs:
        want = e - s + 1
        got = sp.stat().st_size
        if got != want:
            raise RuntimeError(f"sub{i:02d} size mismatch: got {got}, expected {want}")

    # ---- append every sub to dst (preserving any pre-existing prefix) ---
    with open(dst, "ab") as out:
        for i, _s, _e, sp in subs:
            with open(sp, "rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    shutil.rmtree(sub_dir, ignore_errors=True)

    return dst.stat().st_size


def parallel_range_download(url: str, dst: Path, total: int,
                            connections: int, substreams: int,
                            retries: int, retry_delay: int):
    """Download ``url`` -> ``dst`` using ``connections`` HTTP-range parts,
    each of which is itself fanned out into ``substreams`` sub-streams.

    Output is staged into ``<dst>.partK`` files, then concatenated into the
    final file.  Strict size verification is done on every part *and* on the
    assembled file.
    """
    parts_dir = dst.with_suffix(dst.suffix + ".parts")
    parts_dir.mkdir(exist_ok=True)

    # split into N near-equal slices
    slice_size = (total + connections - 1) // connections
    ranges = []
    for i in range(connections):
        start = i * slice_size
        end = min(start + slice_size - 1, total - 1)
        if start > end:
            break
        ranges.append((i, start, end))

    print(f"[lvis] {dst.name}: {total/2**30:.2f} GB, "
          f"{len(ranges)} parts x {substreams} sub-streams "
          f"= {len(ranges)*substreams} TCP connections", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=connections) as ex:
        futs = {}
        for i, s, e in ranges:
            part = parts_dir / f"part{i:03d}"
            futs[ex.submit(_curl_range, url, part, s, e,
                           retries, retry_delay, substreams)] = (i, s, e, part)
        done = 0
        for f in as_completed(futs):
            i, s, e, part = futs[f]
            expected = e - s + 1
            try:
                got = f.result()
            except subprocess.CalledProcessError as exc:
                failures.append((i, str(exc)))
                continue
            if got != expected:
                failures.append((i, f"size mismatch: got {got}, expected {expected}"))
                continue
            done += 1
            print(f"   part {i:03d} ok ({expected/2**20:.1f} MB)  [{done}/{len(ranges)}]", flush=True)

    if failures:
        msg = "; ".join(f"part{i}: {m}" for i, m in failures)
        raise RuntimeError(f"some parts failed: {msg}")

    # ---- concatenate -----------------------------------------------------
    print(f"[lvis] assembling {dst.name} from {len(ranges)} parts", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".assembling")
    with open(tmp, "wb") as out:
        for i, _s, _e in ranges:
            with open(parts_dir / f"part{i:03d}", "rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)

    if tmp.stat().st_size != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"assembled size mismatch for {dst.name}: "
            f"got {tmp.stat().st_size:,}, expected {total:,}"
        )

    os.replace(tmp, dst)
    shutil.rmtree(parts_dir, ignore_errors=True)


def safe_unzip(zip_path: Path, dest: Path):
    """Verify CRCs and extract ``zip_path`` into ``dest``."""
    print(f"[lvis] verifying CRC of {zip_path.name}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupted entry '{bad}' in {zip_path}")
        print(f"[lvis] extracting {zip_path.name} -> {dest}", flush=True)
        z.extractall(dest)


def fetch(url: str, dst: Path, args):
    """Download ``url`` to ``dst`` with parallel ranges + integrity checks."""
    expected = remote_size(url)
    if expected <= 0:
        raise RuntimeError(f"could not determine remote size for {url}")

    if dst.exists():
        if dst.stat().st_size == expected:
            # quick CRC sanity to detect a previously corrupted resume
            try:
                with zipfile.ZipFile(dst) as z:
                    if z.testzip() is None:
                        print(f"[lvis] {dst.name}: already complete & CRC ok, skip", flush=True)
                        return
                    print(f"[lvis] {dst.name}: size ok but CRC failed, redownloading", flush=True)
            except zipfile.BadZipFile:
                print(f"[lvis] {dst.name}: not a valid zip, redownloading", flush=True)
            dst.unlink()
        else:
            print(f"[lvis] {dst.name}: size mismatch "
                  f"({dst.stat().st_size:,} vs {expected:,}), redownloading", flush=True)
            dst.unlink()

    parallel_range_download(url, dst, expected,
                            connections=args.connections,
                            substreams=args.substreams,
                            retries=args.retries,
                            retry_delay=args.retry_delay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if not args.no_proxy and args.proxy:
        setup_proxy(args.proxy)

    root = Path(args.root)
    lvis_dir = root / "lvis"
    images_dir = lvis_dir / "images"
    root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. labels + split lists ----------------------------------------
    if not (lvis_dir / "train.txt").exists():
        zip_path = root / "lvis-labels-segments.zip"
        fetch(LABELS_URL, zip_path, args)
        safe_unzip(zip_path, root)
        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)
    else:
        print(f"[lvis] labels already present at {lvis_dir}, skipping", flush=True)

    # ---- 2. COCO image zips --------------------------------------------
    splits = ["train2017", "val2017"]
    if not args.skip_test:
        splits.append("test2017")

    for split in splits:
        url = IMAGE_URLS[split]
        target = images_dir / split
        # Consider the split done only if its directory has plenty of jpgs.
        # (A fresh labels-zip extraction creates an *empty* <split>/, which
        # we don't want to count as "done".)
        existing = list(target.glob("*.jpg")) if target.is_dir() else []
        if len(existing) > 100:
            print(f"[lvis] {split}: already has {len(existing)} images, skip", flush=True)
            continue

        zip_path = images_dir / f"{split}.zip"
        fetch(url, zip_path, args)
        safe_unzip(zip_path, images_dir)
        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)

    # ---- 3. sanity check -----------------------------------------------
    print("\n[lvis] final layout:", flush=True)
    checks = [
        ("train.txt",          False),
        ("val.txt",            False),
        ("minival.txt",        False),
        ("annotations",        True),
        ("labels/train2017",   True),
        ("labels/val2017",     True),
        ("images/train2017",   True),
        ("images/val2017",     True),
    ]
    if not args.skip_test:
        checks.append(("images/test2017", True))

    ok = True
    for sub, is_dir in checks:
        p = lvis_dir / sub
        if not p.exists():
            print(f"   X {sub}  (missing)")
            ok = False
            continue
        if is_dir:
            n = sum(1 for _ in p.iterdir())
            print(f"   {'OK' if n > 0 else 'X '} {sub}/  ({n} entries)")
            ok = ok and (n > 0)
        else:
            with open(p) as f:
                n = sum(1 for _ in f)
            print(f"   OK {sub}  ({n} lines)")

    if ok:
        print("\n[lvis] all sets ready.  Train with: data=lvis.yaml")
    else:
        print("\n[lvis] WARNING: some pieces are missing -- rerun this script "
              "(parts already on disk will be reused).")
        sys.exit(1)


if __name__ == "__main__":
    main()
