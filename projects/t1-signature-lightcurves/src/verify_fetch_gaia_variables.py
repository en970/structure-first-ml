"""Fault-injection checks on the two safety mechanisms in fetch_gaia_variables.py.

A check that cannot fire is worse than no check, so both mechanisms are driven with an
injected fault and are required to react. Neither check touches the network: the Gaia
module is stubbed, so the only thing under test is the logic in the fetcher.

  CHECK 1  The per-class retrieval-rate warning. Fed a cache in which 45 of 50 CEP objects
           are withheld and every other class is complete, it must print WARNING and report
           a spread near 0.9. WHAT WOULD MAKE IT FAIL: computing the spread over the wrong
           axis, or comparing against the fetched set rather than the requested sample --
           either would report 0.0 and stay silent while one class had been destroyed. The
           negative control feeds the same function the unbiased cache and requires silence,
           so a warning that always fires also fails.

  CHECK 2  The batch-halving fallback. One identifier in a 25-id batch is poisoned so that
           any call containing it raises. All 24 healthy sources must be recovered. WHAT
           WOULD MAKE IT FAIL: no fallback at all (0 of 25 recovered), or a depth cap that
           abandons a pair once it is reached. The first draft of the fetcher had exactly
           that cap and this check recovered 23, not 24, which is how the defect was found.

Requires a populated `data/gaia_epoch_cache` and `data/gaia_dr3_vari_pool_pilot.csv`, i.e.
`python3 src/fetch_gaia_variables.py --per-class 50 --tag pilot` must have been run once.

Run:  python3 src/verify_fetch_gaia_variables.py     # 3/3 checks
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import fetch_gaia_variables as F  # noqa: E402

POOL = ROOT / "data" / "gaia_dr3_vari_pool_pilot.csv"
CACHE = ROOT / "data" / "gaia_epoch_cache"


class _FakeTable:
    """Minimal stand-in for astropy.table.Table as fetch_gaia_variables consumes it."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


def _install_fake_gaia(poison: int, calls: dict) -> None:
    class _FakeGaia:
        ROW_LIMIT = -1

        @staticmethod
        def load_data(ids, **_kw):
            calls["n"] = calls.get("n", 0) + 1
            if poison in ids:
                raise RuntimeError("injected HTTP 500 for the poisoned identifier")
            out = {}
            for sid in ids:
                p = CACHE / f"{sid}.parquet"
                if p.exists():
                    out[f"EPOCH_PHOTOMETRY-Gaia DR3 {sid}.csv"] = [
                        _FakeTable(pd.read_parquet(p))]
            return out

    mod = types.ModuleType("astroquery.gaia")
    mod.Gaia = _FakeGaia
    sys.modules["astroquery.gaia"] = mod


def main() -> int:
    if not POOL.exists() or not any(CACHE.glob("*.parquet")):
        print(f"missing pilot data; run first:\n"
              f"  python3 src/fetch_gaia_variables.py --per-class 50 --tag pilot",
              file=sys.stderr)
        return 2

    with contextlib.redirect_stdout(io.StringIO()):
        sample = F.select_sample(pd.read_csv(POOL), 50)

    results: list[tuple[str, bool, str]] = []
    tmp = Path(tempfile.mkdtemp(prefix="verify_gaia_"))
    try:
        # ---- CHECK 1: the warning must fire on class-correlated attrition -------------
        cep = [int(s) for s in sample.loc[sample.best_class_name == "CEP", "source_id"]]
        withheld = set(cep[:45])
        biased = tmp / "biased"
        biased.mkdir()
        for p in CACHE.glob("*.parquet"):
            if int(p.stem) not in withheld:
                shutil.copy(p, biased / p.name)
        for sid in withheld:
            (biased / f"{sid}.empty").write_text("")   # negative cache: no refetch attempted

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, stats = F.fetch_epoch_photometry(sample, workers=1, batch=25, cache_dir=biased)
        fired = "WARNING" in buf.getvalue()
        spread = stats["retrieval_rate_spread"]
        results.append(("class-attrition warning fires on biased attrition",
                        fired and spread > 0.8, f"spread={spread}, warning={fired}"))

        # ---- CHECK 1b: and must stay silent when attrition is absent ------------------
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, stats_ok = F.fetch_epoch_photometry(sample, workers=1, batch=25, cache_dir=CACHE)
        silent = "WARNING" not in buf.getvalue()
        results.append(("class-attrition warning silent on unbiased cache",
                        silent and stats_ok["retrieval_rate_spread"] < 0.15,
                        f"spread={stats_ok['retrieval_rate_spread']}, silent={silent}"))

        # ---- CHECK 2: batch halving must isolate the single poisoned identifier -------
        ids = [int(s) for s in sample.source_id[:25]]
        calls: dict = {}
        _install_fake_gaia(ids[7], calls)
        halving = tmp / "halving"
        halving.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            ok, bad = F._fetch_batch(ids, halving, retries=1)
        recovered = len(list(halving.glob("*.parquet")))
        results.append(("batch halving isolates one poisoned id",
                        recovered == 24 and ok == 24 and bad == 1,
                        f"recovered={recovered}/25, ok={ok}, bad={bad}, "
                        f"load_data calls={calls.get('n')}"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.modules.pop("astroquery.gaia", None)

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    print(f"{n_pass}/{len(results)} checks pass")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
