# Data sources and access recipes

*Compiled 26 July 2026. Every recipe below was tested live during the scan unless marked
otherwise; the result of that test is stated, including the failures.*

---

## ZTF Bright Transient Survey — labels

The BTS sample explorer publishes a flat CSV of spectroscopically classified ZTF transients, with
no authentication.

```
https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&format=csv
```

**Tested:** HTTP 200, 20,546 rows, of which **12,916 carry a spectroscopic classification** (the
remainder are still candidates, marked `-`). Class counts in the classified subset: SN Ia 7,894,
SN II 1,736, CV 685, AGN 404, SN IIn 302, SN Ia-91T 262, SN Ic 224, SN Ib 191, SLSN-I 111, and a
tail of rarer types.

The explorer page carries thumbnail plots only; the per-epoch photometry has to be fetched
separately, which is what the next entry is for.

## ALeRCE broker — ZTF per-epoch photometry

The ALeRCE ZTF API serves individual detections with no API key.

```python
import requests

oid = "ZTF18abbuksn"
meta = requests.get(f"https://api.alerce.online/ztf/v1/objects/{oid}", timeout=30).json()
det = requests.get(f"https://api.alerce.online/ztf/v1/objects/{oid}/detections", timeout=30).json()
# each detection: {"mjd": ..., "fid": 1|2, "magpsf": ..., "sigmapsf": ..., ...}
# fid 1 = g, fid 2 = r
```

**Tested:** 1,044 detections returned for the object above, `fid` taking values in `{1, 2}`, no
authentication required. This yields exactly the $(t, m, \sigma, \text{band})$ tuples the signature
construction consumes, with no interpolation anywhere in the path.

ALeRCE additionally serves its own light-curve-classifier labels through `query_objects`, keyed on
the same object identifiers. Those are machine labels and are kept strictly separate from the BTS
spectroscopic labels, which are the ground truth used here.

## IRSA ZTF Data Release light curves

```
https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID=<oid>&FORMAT=CSV
```

**Tested:** `ID=686103400067717&FORMAT=CSV` returned HTTP 200 in about 4 s with 849 rows and the
columns `oid, expid, hjd, mjd, mag, magerr, catflags, filtercode, ra, dec`. Confirmed real,
single-band, irregular cadence.

**Failure to record:** cone-search queries of the form `POS=CIRCLE ra dec radius` timed out on three
attempts (45 s, 60 s, 90 s) from this environment, while the IRSA host itself responded in under a
second. IRSA's own documentation directs bulk users to the HATS parquet catalogue rather than the
CGI cone search. Identifier-based queries are therefore the supported route for assembling a sample,
and the cone-search path should not be relied on.

The API's default collection is `ztf_dr18`; later collections are selected with `COLLECTION=ztf_dr23`
and similar. The survey itself has released through DR24 **[reported]**.

## Gaia DR3 epoch photometry — the counter-test sample

```python
from astroquery.gaia import Gaia

job = Gaia.launch_job_async(
    "SELECT source_id, best_class_name, best_class_score "
    "FROM gaiadr3.vari_classifier_result "
    "WHERE best_class_score > 0.9"
)
```

**Tested:** the classifier table returns **9,976,881** classified variable sources with
`best_class_name` labels (ECL, RS, SOLAR_LIKE and the rest of the DR3 variability taxonomy).

Per-epoch photometry comes from DATALINK:

```
RETRIEVAL_TYPE=EPOCH_PHOTOMETRY&DATA_STRUCTURE=INDIVIDUAL&FORMAT=CSV
```

**Tested:** returned a ZIP of per-source CSVs, 37 to 48 transit rows each, with columns
`g_transit_time/flux/flux_error`, `bp_obs_time/flux/flux_error`, `rp_obs_time/flux/flux_error`.
Measured at roughly **6.1 kB per source compressed**, so 50,000 sources is about 0.3 GB — trivially
laptop-scale. Two practical constraints: DATALINK caps a call at 5,000 source identifiers, so a
large sample needs sequential batches; and the format is wide, one row per transit with a separate
timestamp for each band, so it must be reshaped to long $(t, \text{flux}, \sigma, \text{band})$
tuples. The per-band timestamps differ within a transit, which is a genuine feature of the data
rather than an inconvenience — the three bands really are sampled at different instants.

## Datasets considered and set aside

**Kepler, K2 and TESS.** Confirmed near-regular cadence: Kepler at 29.4 min or 1 min, TESS at 2 min,
20 s or 30 min within a sector, with gaps at data downlink. These are excellent data and the wrong
data for a study whose subject is irregular sampling. Set aside for that reason, not for quality.

**PLAsTiCC and ELAsTiCC.** Simulated LSST-like light curves. PLAsTiCC remains available on Zenodo;
ELAsTiCC is the benchmark brokers currently test against. The community norm is to use simulated and
real data together rather than simulated alone. Held in reserve as a possible extension, since
simulation allows the sampling ablation to be driven from known ground truth.

**Chen et al. (2020) ZTF periodic-variable catalogue.** 781,602 sources in 11 classes, on VizieR at
`J/ApJS/249/18`. The catalogue itself carries periods and features but not per-epoch photometry,
which would have to be pulled through the IRSA path above. A strong option for a periodic-variable
extension of the counter-test.

---

## What T1 uses, and why both samples are needed

The two primary samples are chosen to test opposite halves of the same prediction.

**ZTF BTS transients — where signatures should win.** Supernova classes are distinguished largely by
the *shape and ordering* of the light curve: rise time against decline rate, the presence and
position of a secondary maximum, the asymmetry between brightening and fading. Ordering is precisely
what iterated integrals encode and what order-blind summary statistics discard. The sampling is
genuinely irregular, driven by weather and survey scheduling, and light curves are short and sparse.

**Gaia DR3 variables — where signatures should lose.** Periodic variables are distinguished by
period and amplitude. Period is a statement about the clock, and the signature is invariant to
reparameterisation of the clock, so a plain signature should discard exactly the information that
matters. Recovering competitive performance should require explicitly adding a time channel, and the
size of that gap is a direct measurement of how much the invariance costs when it is the wrong
invariance.

A method that wins on both samples for the same reasons has not been understood. The interesting
outcome is the asymmetry, and if it fails to appear the explanation offered in T1's design is wrong.
