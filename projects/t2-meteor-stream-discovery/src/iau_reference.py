"""The IAU Meteor Data Center shower lists, as the reference against which claims are made.

Two lists matter and they mean different things:

  established  Showers the IAU has formally accepted. These are the positive controls: a
               method that cannot recover them has no standing to propose anything.
  working      Candidate showers submitted but not yet established. These are the far more
               dangerous list. A group that matches a working-list entry is NOT a
               discovery -- someone has already reported it -- and checking only the
               established list would manufacture false novelty.

Both are downloaded from the MDC directly and cached. The format is a pipe-separated
fixed-width table with roughly ninety-eight comment lines prefixed by a colon.

Proper citation for the database, per the MDC's own instructions:
  Jenniskens, P.; Jopek, T.J.; Janches, D.; Hajdukova, M.; Kokhirova, G.I.; Rudawska, R.
  (2020), Planetary and Space Science, Vol. 182, article id. 104821.
  Jopek, T.J.; Kanuchova, Z. (2017), Planetary and Space Science, Vol. 143, p. 3-6.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.ta3.sk/IAUC22DB/MDC2022/Etc/"
FILES = {"established": "streamestablisheddata2026.txt",
         "working": "streamworkingdata2026.txt"}

# Field order taken from the MDC header block. Only the columns used downstream are
# named; the remainder are kept positionally and ignored.
FIELDS = ["lp", "iau_no", "ad_no", "code", "status", "sub_date", "name", "activity",
          "lo_s_begin", "lo_s_end", "lo_s_max", "ra", "dec", "d_ra", "d_dec", "vg",
          "lo_r"]

__all__ = ["load_iau_lists", "match_to_iau", "ANGULAR_TOL_DEG", "VG_TOL_FRAC",
           "SOLLON_TOL_DEG"]

# Matching tolerances. Deliberately generous: the cost of wrongly calling a group novel
# (a false discovery claim) is far higher than the cost of wrongly calling it known
# (a missed candidate that a stricter pass would recover).
ANGULAR_TOL_DEG = 8.0    # radiant separation
VG_TOL_FRAC = 0.15       # fractional geocentric-velocity agreement
SOLLON_TOL_DEG = 12.0    # solar-longitude (activity-time) agreement


def _download(name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / FILES[name]
    if out.exists() and out.stat().st_size > 10_000:
        return out
    resp = requests.get(BASE + FILES[name], timeout=120, allow_redirects=True)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return out


def _parse(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(":") or line.startswith("+") or not line.strip():
            continue
        parts = [p.strip().strip('"').strip() for p in line.split("|")]
        if len(parts) < len(FIELDS):
            continue
        rows.append(parts[:len(FIELDS)])
    df = pd.DataFrame(rows, columns=FIELDS)
    for col in ("lo_s_begin", "lo_s_end", "lo_s_max", "ra", "dec", "vg"):
        df[col] = pd.to_numeric(df[col].replace({"": np.nan}), errors="coerce")
    df["code"] = df["code"].str.strip()
    df["name"] = df["name"].str.strip()
    return df[df.code.str.len() == 3]


def load_iau_lists(cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Download (once) and parse both MDC lists."""
    out = {}
    for name in FILES:
        df = _parse(_download(name, cache_dir))
        out[name] = df
    return out


def _angsep(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees, vectorised over the second argument."""
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2), np.radians(dec2)
    cos = (np.sin(d1) * np.sin(d2)
           + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def match_to_iau(ra: float, dec: float, vg: float, sol_lon: float,
                 lists: dict[str, pd.DataFrame]) -> dict:
    """Match one candidate against both MDC lists on radiant, speed and activity time.

    Returns the best match found in each list, or None. A candidate is only reportable as
    novel when BOTH come back empty.
    """
    result = {}
    for name, df in lists.items():
        ok = df.dropna(subset=["ra", "dec", "vg"])
        if ok.empty:
            result[name] = None
            continue
        sep = _angsep(ra, dec, ok.ra.to_numpy(), ok.dec.to_numpy())
        vg_ok = np.abs(ok.vg.to_numpy() - vg) <= VG_TOL_FRAC * max(vg, 1e-6)

        # Activity time: use the listed maximum where present, otherwise accept.
        lo = ok.lo_s_max.to_numpy()
        dsol = np.abs((lo - sol_lon + 180.0) % 360.0 - 180.0)
        sol_ok = ~np.isfinite(lo) | (dsol <= SOLLON_TOL_DEG)

        cand = (sep <= ANGULAR_TOL_DEG) & vg_ok & sol_ok
        if not cand.any():
            result[name] = None
            continue
        best = int(np.argmin(np.where(cand, sep, np.inf)))
        result[name] = {
            "code": ok.iloc[best]["code"],
            "name": ok.iloc[best]["name"],
            "separation_deg": round(float(sep[best]), 2),
            "vg_listed": float(ok.iloc[best]["vg"]),
            "lo_s_max_listed": (float(lo[best]) if np.isfinite(lo[best]) else None),
        }
    return result
