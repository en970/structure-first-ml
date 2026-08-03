#!/usr/bin/env python3
"""Reclaim disk from regenerable download caches, without touching results.

The pipelines here cache aggressively on purpose: re-running a fetch should be cheap, and
per-object caches make a resumed download nearly free. The cost is that those caches grow
without bound. On 2026-08-04 they filled the disk to the point where no shell command could
run at all, which stops work completely.

The distinction this script enforces:

  SAFE TO DELETE   per-object or per-file download caches. Every one of them is rebuilt
                   automatically by the fetch script that made it, and none is an input to
                   any result that is not also stored in an assembled file.

  NEVER DELETE     assembled datasets (the parquet files the analyses actually read),
                   reference catalogues that are small and awkward to refetch, everything
                   under outputs/, and anything tracked by git.

Deleting a cache costs a re-download, not a result. Deleting an assembled parquet costs
hours. The lists below encode that difference; do not move an entry between them without
checking which fetch script rebuilds it.

Usage:
  python3 tools/cleanup_caches.py            # report only, delete nothing
  python3 tools/cleanup_caches.py --delete   # actually reclaim
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories rebuilt automatically by a fetch script. The comment names the rebuilder.
REGENERABLE = [
    ("projects/t1-signature-lightcurves/data/alerce_cache", "src/fetch_ztf_bts.py"),
    ("projects/t1-signature-lightcurves/data/gaia_epoch_cache", "src/fetch_gaia_variables.py"),
    ("projects/t2-meteor-stream-discovery/data/gmn_monthly", "src/fetch_gmn.py"),
    ("projects/t4-plate-topology/data/dasch_cache", "src/dasch_access.py"),
]

# Kept even though they are technically re-fetchable: assembling them is slow, they are the
# direct inputs to published numbers, and they are small enough to be worth the space.
PROTECTED = [
    "projects/t1-signature-lightcurves/data/ztf_bts_lightcurves.parquet",
    "projects/t1-signature-lightcurves/data/gaia_dr3_variables_full.parquet",
    "projects/t2-meteor-stream-discovery/data/gmn_orbits.parquet",
    "projects/t2-meteor-stream-discovery/data/iau",
]


def size_of(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def tracked_by_git(path: Path) -> bool:
    """Refuse to delete anything git knows about, whatever the lists say."""
    try:
        out = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                             cwd=ROOT, capture_output=True, text=True, timeout=20)
        return out.returncode == 0
    except Exception:
        return True  # if the check itself fails, assume tracked and keep the file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="actually remove; without this the script only reports")
    args = ap.parse_args()

    print(f"repository: {ROOT}\n")
    print("regenerable caches")
    total = 0
    targets = []
    for rel, rebuilder in REGENERABLE:
        p = ROOT / rel
        s = size_of(p)
        total += s
        state = "absent" if not p.exists() else human(s)
        print(f"  {rel:58s} {state:>10s}   rebuilt by {rebuilder}")
        if p.exists() and s > 0:
            targets.append(p)

    print(f"\nreclaimable: {human(total)}")

    print("\nprotected (assembled inputs to published numbers, never auto-deleted)")
    for rel in PROTECTED:
        p = ROOT / rel
        print(f"  {rel:58s} {human(size_of(p)):>10s}")

    if not args.delete:
        print("\nreport only. re-run with --delete to reclaim.")
        return 0

    freed = 0
    for p in targets:
        if tracked_by_git(p):
            print(f"  SKIP (tracked by git): {p.relative_to(ROOT)}")
            continue
        s = size_of(p)
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            freed += s
            print(f"  removed {p.relative_to(ROOT)} ({human(s)})")
    print(f"\nfreed {human(freed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
