"""Orbital dissimilarity criteria: D_SH, D_D, D_H, and D_N.

Meteoroid-stream identification is a clustering problem in which the metric is not a free
choice: two meteoroids belong to the same stream when their heliocentric orbits are
similar, and "similar orbit" is a statement about celestial mechanics. The classical
criteria implemented here are competing formalisations of that statement, and they
disagree with each other in ways that propagate directly into which streams get found.
That disagreement is the point of this track: a candidate that survives all of them is
worth attention, one that appears under a single metric is an artefact of that metric.

Implemented, with original references:

  D_SH  Southworth & Hawkins (1963). Terms in (e, q, mutual inclination, perihelion
        orientation), all absolute.
  D_D   Drummond (1981). Normalised e and q terms, angles in radians scaled by pi,
        perihelion-point angle instead of the Southworth-Hawkins longitude term.
  D_H   Jopek (1993). Hybrid: the Southworth-Hawkins form with Drummond's normalised
        q term.
  D_N   Valsecchi, Jopek & Froeschle (1999). Built from near-invariants of the
        encounter geometry (U, cos theta) rather than osculating elements, so it is
        less sensitive to secular evolution of the angular elements.

Conventions: angles in DEGREES at the interface, radians internally; q in AU; U in units
of Earth's mean orbital speed. All functions are vectorised: given arrays of shape (n,)
and (m,) they broadcast to an (n, m) block, which is how the pairwise computations in the
clustering stage are built without a Python loop.

Verification is in `verify_dcriteria.py`: identity, symmetry, and analytic special cases
in which each formula collapses to a closed form.
"""
from __future__ import annotations

import numpy as np

__all__ = ["d_sh", "d_d", "d_h", "d_n", "CRITERIA"]

_D2R = np.pi / 180.0
EARTH_V = 29.784  # km/s, mean Earth orbital speed, for U = vgeo / EARTH_V


def _mutual_inclination_term(i1, i2, node1, node2):
    """(2 sin(I21/2))^2 for the mutual inclination I21 between two orbit planes."""
    return ((2.0 * np.sin(0.5 * (i2 - i1))) ** 2
            + np.sin(i1) * np.sin(i2) * (2.0 * np.sin(0.5 * (node2 - node1))) ** 2)


def _pi21_term(i1, i2, node1, node2, peri1, peri2, sinI2_half_sq):
    """(2 sin(Pi21/2))^2: difference of perihelion longitudes measured from the
    intersection of the two orbital planes (Southworth & Hawkins 1963)."""
    cosI_half = np.sqrt(np.clip(1.0 - 0.25 * sinI2_half_sq, 0.0, 1.0))
    dnode = node2 - node1
    # sign convention: negative branch when |node2 - node1| > 180 deg
    sign = np.where(np.abs(dnode) > np.pi, -1.0, 1.0)
    arg = np.clip(np.cos(0.5 * (i1 + i2)) * np.sin(0.5 * dnode)
                  / np.where(cosI_half == 0.0, np.inf, cosI_half), -1.0, 1.0)
    pi21 = (peri2 - peri1) + sign * 2.0 * np.arcsin(arg)
    return (2.0 * np.sin(0.5 * pi21)) ** 2


def _perihelion_point_angle(i1, i2, node1, node2, peri1, peri2):
    """Angle between the two perihelion directions (Drummond 1981)."""
    sin_b1, sin_b2 = np.sin(i1) * np.sin(peri1), np.sin(i2) * np.sin(peri2)
    b1, b2 = np.arcsin(np.clip(sin_b1, -1, 1)), np.arcsin(np.clip(sin_b2, -1, 1))
    lam1 = node1 + np.arctan2(np.cos(i1) * np.sin(peri1), np.cos(peri1))
    lam2 = node2 + np.arctan2(np.cos(i2) * np.sin(peri2), np.cos(peri2))
    cos_theta = (np.sin(b1) * np.sin(b2)
                 + np.cos(b1) * np.cos(b2) * np.cos(lam1 - lam2))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def _bc(x1, x2):
    """Broadcast two 1-D arrays to an (n, m) pair block."""
    a1 = np.atleast_1d(np.asarray(x1, dtype=np.float64))[:, None]
    a2 = np.atleast_1d(np.asarray(x2, dtype=np.float64))[None, :]
    return a1, a2


def d_sh(q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2):
    """Southworth & Hawkins (1963). Angles in degrees, q in AU. Returns (n, m)."""
    q1, q2 = _bc(q1, q2)
    e1, e2 = _bc(e1, e2)
    i1, i2 = _bc(np.asarray(i1) * _D2R, np.asarray(i2) * _D2R)
    node1, node2 = _bc(np.asarray(node1) * _D2R, np.asarray(node2) * _D2R)
    peri1, peri2 = _bc(np.asarray(peri1) * _D2R, np.asarray(peri2) * _D2R)

    sinI2 = _mutual_inclination_term(i1, i2, node1, node2)
    piterm = _pi21_term(i1, i2, node1, node2, peri1, peri2, sinI2)
    d2 = ((e2 - e1) ** 2 + (q2 - q1) ** 2 + sinI2
          + (0.5 * (e1 + e2)) ** 2 * piterm)
    return np.sqrt(np.clip(d2, 0.0, None))


def d_d(q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2):
    """Drummond (1981). Angles in degrees, q in AU. Returns (n, m)."""
    q1, q2 = _bc(q1, q2)
    e1, e2 = _bc(e1, e2)
    i1r, i2r = _bc(np.asarray(i1) * _D2R, np.asarray(i2) * _D2R)
    node1r, node2r = _bc(np.asarray(node1) * _D2R, np.asarray(node2) * _D2R)
    peri1r, peri2r = _bc(np.asarray(peri1) * _D2R, np.asarray(peri2) * _D2R)

    sinI2 = _mutual_inclination_term(i1r, i2r, node1r, node2r)
    I21 = 2.0 * np.arcsin(np.clip(0.5 * np.sqrt(sinI2), -1.0, 1.0))
    theta = _perihelion_point_angle(i1r, i2r, node1r, node2r, peri1r, peri2r)

    d2 = (((e2 - e1) / (e1 + e2 + 1e-300)) ** 2
          + ((q2 - q1) / (q1 + q2 + 1e-300)) ** 2
          + (I21 / np.pi) ** 2
          + (0.5 * (e1 + e2)) ** 2 * (theta / np.pi) ** 2)
    return np.sqrt(np.clip(d2, 0.0, None))


def d_h(q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2):
    """Jopek (1993) hybrid. Angles in degrees, q in AU. Returns (n, m)."""
    q1, q2 = _bc(q1, q2)
    e1, e2 = _bc(e1, e2)
    i1r, i2r = _bc(np.asarray(i1) * _D2R, np.asarray(i2) * _D2R)
    node1r, node2r = _bc(np.asarray(node1) * _D2R, np.asarray(node2) * _D2R)
    peri1r, peri2r = _bc(np.asarray(peri1) * _D2R, np.asarray(peri2) * _D2R)

    sinI2 = _mutual_inclination_term(i1r, i2r, node1r, node2r)
    piterm = _pi21_term(i1r, i2r, node1r, node2r, peri1r, peri2r, sinI2)
    d2 = ((e2 - e1) ** 2
          + ((q2 - q1) / (q1 + q2 + 1e-300)) ** 2
          + sinI2
          + (0.5 * (e1 + e2)) ** 2 * piterm)
    return np.sqrt(np.clip(d2, 0.0, None))


def d_n(u1, ct1, phi1, lam1, u2, ct2, phi2, lam2, w1=1.0, w2=1.0, w3=1.0):
    """Valsecchi, Jopek & Froeschle (1999), geocentric-invariant criterion.

    Inputs: U (geocentric speed / Earth orbital speed), cos(theta), and the angles phi
    and lambda (degrees) of the encounter geometry. The angular term takes the minimum
    over the two symmetry branches, as in the original paper.
    """
    u1, u2 = _bc(u1, u2)
    ct1, ct2 = _bc(ct1, ct2)
    phi1, phi2 = _bc(np.asarray(phi1) * _D2R, np.asarray(phi2) * _D2R)
    lam1, lam2 = _bc(np.asarray(lam1) * _D2R, np.asarray(lam2) * _D2R)

    dphi = phi2 - phi1
    dlam = lam2 - lam1
    branch_1 = w2 * (2.0 * np.sin(0.5 * dphi)) ** 2 + w3 * (2.0 * np.sin(0.5 * dlam)) ** 2
    branch_2 = (w2 * (2.0 * np.sin(0.5 * (dphi + np.pi))) ** 2
                + w3 * (2.0 * np.sin(0.5 * (dlam + np.pi))) ** 2)
    dxi2 = np.minimum(branch_1, branch_2)

    d2 = (u2 - u1) ** 2 + w1 * (ct2 - ct1) ** 2 + dxi2
    return np.sqrt(np.clip(d2, 0.0, None))


def geocentric_invariants(vgeo_kms, a_au, lamgeo_deg, betgeo_deg, sol_lon_deg):
    """(U, cos theta, phi, lambda) from GMN columns, after Valsecchi et al. (1999).

    U is the geocentric speed in units of Earth's orbital speed. cos(theta) is the angle
    between the geocentric velocity and the direction of Earth's motion (the apex),
    computed from the ecliptic radiant: theta is the elongation of the geocentric radiant
    from the apex, whose ecliptic longitude is sol_lon - 90 deg. phi is the position
    angle of the radiant about the apex direction.
    """
    vgeo = np.asarray(vgeo_kms, dtype=np.float64)
    lam = np.asarray(lamgeo_deg, dtype=np.float64) * _D2R
    bet = np.asarray(betgeo_deg, dtype=np.float64) * _D2R
    sol = np.asarray(sol_lon_deg, dtype=np.float64) * _D2R

    u = vgeo / EARTH_V
    # apex of Earth's way: ecliptic longitude = solar longitude - 90 deg, latitude 0.
    dl = lam - (sol - 0.5 * np.pi)
    cos_theta = np.cos(bet) * np.cos(dl)
    # the radiant points where the meteoroid comes FROM; velocity is opposite, so the
    # angle between the geocentric velocity and the apex has cosine -cos_theta.
    cos_theta = -np.clip(cos_theta, -1.0, 1.0)
    phi = np.arctan2(np.sin(bet), np.cos(bet) * np.sin(dl)) / _D2R
    lam_out = np.asarray(lamgeo_deg, dtype=np.float64)
    return u, cos_theta, phi % 360.0, lam_out


CRITERIA = {"d_sh": d_sh, "d_d": d_d, "d_h": d_h}
