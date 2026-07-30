"""Access the DASCH photographic plate archive as IMAGES rather than as light curves.

The Harvard plate collection covers the sky from the 1880s to the 1990s: 429,274 plates,
digitisation completed in 2024. It carries something no modern survey has -- a century-long
time axis on the same patches of sky.

The opening this track works is narrow and evidenced. Of twenty arXiv papers using DASCH,
eighteen extract point-source light curves and the remaining two touch plate imagery only
inside the reduction pipeline's own defect handling. Nobody appears to treat the plates as
2-D images, or a plate series as an (x, y, epoch) object, and analyse their structure
directly.

Endpoints below were verified live on 2026-07-30 against
https://api.starglass.cfa.harvard.edu/public with no authentication and no institutional
account:

  POST /plates/search                 coordinate search; returned 9,036 plates near one
                                      position
  GET  /plates/p/{plate_id}           metadata; e.g. a03393 is a 1898-11-11 exposure at
                                      35268 x 31627 pixels full resolution
  GET  /plates/p/{id}/mosaic          presigned FITS URL; bin_factor=16 gives 4.8 MB and a
                                      (1976, 2204) int16 image with real RA/DEC---TAN WCS,
                                      bin_factor=01 gives 1.24 GB
  POST /dasch/dr7/cutout              arbitrary-position calibrated cutout, returned as
                                      gzip+base64 FITS

Sizes matter for what is feasible: binned mosaics are megabytes, so a working set of
hundreds of plates fits on a laptop, and full-resolution scans are avoided entirely.

Licence: the archive's pages link to Harvard's terms of use. Those terms were NOT read
during the access test and are recorded here as unverified. They must be checked before any
plate imagery is redistributed; nothing in this repository redistributes plate pixels.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

BASE = "https://api.starglass.cfa.harvard.edu/public"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "dasch_cache"

__all__ = ["search_plates", "plate_metadata", "binned_mosaic", "cutout",
           "exposures", "plate_epoch", "has_mosaic", "CACHE"]


def _get(path: str, **params):
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}{path}", params=params or None, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2.0 * (attempt + 1))
    return None


def _post(path: str, payload: dict):
    for attempt in range(4):
        try:
            r = requests.post(f"{BASE}{path}", json=payload, timeout=180)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2.0 * (attempt + 1))
    return None


def search_plates(ra_deg: float, dec_deg: float, radius_arcmin: float = 600,
                  precision: str = "low") -> list[str]:
    """Plate identifiers covering a sky position.

    The endpoint returns a list of plain identifier STRINGS, not records -- verified
    against a real call, which returned 9,036 strings such as 'a02042'. Metadata requires a
    second call per plate.
    """
    out = _post("/plates/search", {"coordinates": {
        "ra": ra_deg, "dec": dec_deg, "precision": precision,
        "radius": radius_arcmin}})
    plates = (out or {}).get("plates", []) or []
    return [p if isinstance(p, str) else p.get("plate_id") for p in plates]


def plate_metadata(plate_id: str) -> dict | None:
    """Plate record: exposures, dates, astrometric solutions, dimensions."""
    return _get(f"/plates/p/{plate_id}")


def binned_mosaic(plate_id: str, bin_factor: int = 16,
                  cache: Path = CACHE) -> tuple[np.ndarray, fits.Header] | None:
    """Whole-plate image at reduced resolution, cached on disk.

    bin_factor 16 was measured at 4.8 MB and (1976, 2204) pixels, which is the right scale
    for structural analysis: large enough to carry morphology, small enough that a few
    hundred plates stay on a laptop.
    """
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{plate_id}_mosaic{bin_factor:02d}.fits.fz"
    if not local.exists():
        info = _get(f"/plates/p/{plate_id}/mosaic", bin_factor=f"{bin_factor:02d}")
        if not info or "presigned_link" not in info:
            return None
        try:
            blob = requests.get(info["presigned_link"], timeout=300).content
        except requests.RequestException:
            return None
        local.write_bytes(blob)
    try:
        with fits.open(local) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu is None:
                return None
            return np.asarray(hdu.data, dtype=np.float64), hdu.header
    except Exception:
        local.unlink(missing_ok=True)
        return None


def cutout(plate_id: str, ra_deg: float, dec_deg: float, solution_number: int,
           cache: Path = CACHE) -> tuple[np.ndarray, fits.Header] | None:
    """Calibrated cutout at an arbitrary position, returned as gzip+base64 FITS."""
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{plate_id}_{ra_deg:.4f}_{dec_deg:.4f}_s{solution_number}"
    local = cache / f"{tag}_cutout.fits"
    if not local.exists():
        payload = {"plate_id": plate_id, "center_ra_deg": ra_deg,
                   "center_dec_deg": dec_deg, "solution_number": solution_number}
        raw = _post("/dasch/dr7/cutout", payload)
        if raw is None:
            return None
        try:
            local.write_bytes(gzip.decompress(base64.b64decode(raw)))
        except Exception:
            return None
    try:
        with fits.open(local) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu is None:
                return None
            return np.asarray(hdu.data, dtype=np.float64), hdu.header
    except Exception:
        local.unlink(missing_ok=True)
        return None


def exposures(meta: dict) -> list[dict]:
    """Exposure records. The field is `catalog_exposures`, not `exposures`.

    Each carries ctr_ra, ctr_dec, datetime, exposure_num and exposure_length. Verified
    against a real record: a02042 is a 60-second exposure from 1896-09-04.
    """
    return (meta or {}).get("catalog_exposures", []) or []


def plate_epoch(meta: dict):
    """Observation datetime of the first exposure, as an ISO string, or None."""
    exps = exposures(meta)
    return str(exps[0]["datetime"]) if exps and exps[0].get("datetime") else None


def has_mosaic(meta: dict) -> bool:
    """Whether a whole-plate mosaic exists for this plate.

    Not every plate has one -- a02042 returns an empty `mosaics` list -- so any survey of
    plates must filter on this rather than assume availability.
    """
    return bool((meta or {}).get("mosaics"))


if __name__ == "__main__":
    # Smoke test against the position used in the access verification.
    plates = search_plates(11.446426, -71.535997)
    print(f"search returned {len(plates)} plate ids, e.g. {plates[:3]}")
    n_with = 0
    for pid in plates[:12]:
        meta = plate_metadata(pid)
        if meta is None:
            continue
        mark = "mosaic" if has_mosaic(meta) else "no-mosaic"
        print(f"  {pid}: epoch={plate_epoch(meta)} exposures={len(exposures(meta))} {mark}")
        if has_mosaic(meta) and n_with < 1:
            got = binned_mosaic(pid, 16)
            if got is not None:
                img, hdr = got
                print(f"    -> shape={img.shape} CTYPE1={hdr.get('CTYPE1')} "
                      f"DATE-OBS={hdr.get('DATE-OBS')} "
                      f"range=[{img.min():.0f}, {img.max():.0f}]")
                n_with += 1
