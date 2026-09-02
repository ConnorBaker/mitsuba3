"""The shipped glass albedo tables against an independent implementation of the same model.

WHAT THIS PINS, AND WHY IT EXISTS. `ggx_energy_compensation`'s glass branch reads
`ggx_glass_E` on BOTH sides of the interface instead of `ggx_glass_inv_E` on the exit
side. Half the justification recorded at that line was that the inverse table is "31%
low" at eta 1.45, roughness 0.9, mu 0.9. That measurement integrated the albedo off the
plugin's own `sample`, and that estimator counts an outgoing direction BELOW the horizon
-- `microfacet_shadow_masking` hands `[0, 0, 1]` to `smith_g1`, which defeats the
orientation test that makes `G` vanish there. (Same root cause as the table-generator
defect pinned by `test_ggx_table_provenance.py`; the plugin's eval masks hide it in a
render, an albedo integral has no such mask.)

THE ORACLE. Mitsuba's C++ `roughdielectric` sampled in `TransportMode::Importance`. Its
mean sample weight is the energy albedo by construction, it applies no radiance-
compression factor in that mode, and it shares no line of the Python lobe's eval, pdf,
Fresnel, half-vector or masking code. It is NOT fully independent -- both go through the
same `MicrofacetDistribution` for D and the Smith G1 -- so this pins the estimator and
the tabulation, not the microfacet model itself.

WHAT IT FINDS. At the three points the withdrawn claim quotes, the table is -3.16% /
+6.93% / +25.28% against the oracle: not low by 31%, and HIGH at two of the three. What
the tables do carry is a grazing, high-roughness bias of the SAME sign in both of them
(up to +56%), which is the 1e3 samples per cell they were generated at.

CONTROLS:
  * SUPPORT -- each comparison prints how many cells it rests on and fails at zero.
  * DISCRIMINATION -- the same comparison is run against a deliberately scaled table and
    must REJECT it, so "within tolerance" is not something every input achieves.
  * LOCALITY -- the tables' agreement is asserted where they are good AND the bias is
    asserted to be present where it is bad, so a future regeneration that flattens the
    bias trips this file instead of passing quietly.
"""

import numpy as np
import pytest

import drjit as dr
import mitsuba as mi

N = 1 << 20

# The `precompute` axis: 16 points from 1e-4 to 1 inclusive, on each of r / mu / z.
AXIS = np.linspace(1e-4, 1.0, 16, endpoint=True)


def _variant():
    for v in ("llvm_ad_rgb", "cuda_ad_rgb"):
        if v in mi.variants():
            return v
    pytest.skip("this test integrates millions of BSDF samples; scalar_rgb is unusable")


def _reparam_eta(z):
    """`precompute.reparam_eta`: r0 = z^4, eta = (1 + sqrt(r0)) / (1 - sqrt(r0))."""
    from mitsuba.ad.bsdfs.common import sanitize_eta
    from mitsuba.ad.bsdfs.lobes import r0_to_eta

    return float(sanitize_eta(r0_to_eta(dr.square(dr.square(dr.clip(mi.Float(z), 1e-4, 0.99)))))[0])


def _oracle(roughness, mu, eta, seed=3):
    """Energy albedo of a rough dielectric interface, from the C++ plugin."""
    alpha = max(roughness * roughness, 1e-4)
    if eta >= 1.0:
        props = dict(int_ior=eta, ext_ior=1.0)
    else:
        props = dict(int_ior=1.0, ext_ior=1.0 / eta)
    bsdf = mi.load_dict({"type": "roughdielectric", "distribution": "ggx",
                         "alpha": alpha, **props})

    si = dr.zeros(mi.SurfaceInteraction3f, N)
    si.wi = mi.Vector3f(dr.sqrt(1.0 - mu * mu), 0.0, mu)
    si.sh_frame = mi.Frame3f(mi.Vector3f(0, 0, 1))
    si.wavelengths = mi.Color0f()

    v0, v1 = mi.sample_tea_32(dr.arange(mi.UInt32, N), mi.UInt32(seed))
    rng = mi.PCG32(initstate=v0, initseq=v1)
    sample1 = rng.next_float32()
    sample2 = mi.Point2f(rng.next_float32(), rng.next_float32())

    bs, weight = bsdf.sample(mi.BSDFContext(mi.TransportMode.Importance), si, sample1, sample2)
    w = mi.luminance(weight)
    w = dr.select(dr.isfinite(w) & (bs.pdf > 0.0), w, 0.0)
    return float(dr.sum(w)[0]) / N


def _table(name):
    from mitsuba.ad.bsdfs.tables import fetch_table

    return fetch_table(name)


def _lookup(name, r, mu, z):
    return float(_table(name).eval([mi.Float(r), mi.Float(mu), mi.Float(z)])[0][0])


# The three operating points the withdrawn "31% low" claim quotes.
QUOTED = [(0.9, 0.9), (0.9, 0.5), (0.9, 0.2)]
QUOTED_ETA = 1.45


def _quoted_z(eta_up):
    r0 = ((eta_up - 1.0) / (eta_up + 1.0)) ** 2
    return r0 ** 0.25


def test_glass_inv_table_is_not_31_percent_low():
    """THE REFUTATION. The table must not sit far BELOW the oracle at the quoted points."""
    mi.set_variant(_variant())
    z = _quoted_z(QUOTED_ETA)
    rels = []
    for roughness, mu in QUOTED:
        t = _lookup("ggx_glass_inv_E", roughness, mu, z)
        o = _oracle(roughness, mu, 1.0 / QUOTED_ETA)
        rel = (t - o) / o
        rels.append(rel)
        print(f"r={roughness} mu={mu}: table {t:.6f}  oracle {o:.6f}  {rel:+.2%}")
    assert len(rels) == 3

    worst_low = min(rels)
    assert worst_low > -0.10, (
        "ggx_glass_inv_E really is far below the energy albedo "
        f"(worst {worst_low:+.2%}); the withdrawal in `ggx_energy_compensation` is wrong"
    )
    # And the shape of the disagreement is the OPPOSITE of the withdrawn claim.
    assert max(rels) > 0.10, (
        "the table no longer runs high at grazing -- it has been regenerated, and both "
        "this test and the comment it pins are stale. Re-derive rather than relax."
    )


def test_the_comparison_can_reject():
    """DISCRIMINATION. A table 31% below the oracle must FAIL the check above, or the
    check cannot tell the two states apart and proves nothing about the real one."""
    mi.set_variant(_variant())
    z = _quoted_z(QUOTED_ETA)
    rels = []
    for roughness, mu in QUOTED:
        o = _oracle(roughness, mu, 1.0 / QUOTED_ETA)
        forged = 0.69 * o          # what "31% low" would actually look like
        rels.append((forged - o) / o)
    print(f"forged-31%-low arm: worst {min(rels):+.2%}")
    assert min(rels) <= -0.10, "the -10% bound would admit a table that is 31% low"


# (roughness, mu) cells where 1e3 samples per cell is enough, at the IORs glass uses.
GOOD_CELLS = [(3, 3), (3, 7), (3, 11), (3, 14), (7, 7), (7, 11), (7, 14), (11, 11), (11, 14)]
# ... and where it is not: grazing incidence at high roughness.
BIASED_CELLS = [(11, 3), (13, 3)]
Z_INDICES = [3, 6, 9]           # eta 1.0834 / 1.3811 / 2.1252


@pytest.mark.parametrize("name,invert", [("ggx_glass_E", False), ("ggx_glass_inv_E", True)])
def test_glass_tables_track_the_oracle_away_from_grazing(name, invert):
    """Where the tables are used, they are right -- to 3%."""
    mi.set_variant(_variant())
    tab = np.array(_table(name).tensor())[..., 0]
    rels = []
    for k in Z_INDICES:
        eta_up = _reparam_eta(float(AXIS[k]))
        eta = 1.0 / eta_up if invert else eta_up
        for i, j in GOOD_CELLS:
            r, mu = float(AXIS[i]), float(AXIS[j])
            o = _oracle(r, mu, eta)
            rels.append((float(tab[k, j, i]) - o) / o)
    rels = np.asarray(rels)
    print(f"{name}: {rels.size} cells, mean {rels.mean():+.2%}, max|rel| {np.abs(rels).max():.2%}")
    assert rels.size == len(Z_INDICES) * len(GOOD_CELLS) > 0
    assert np.abs(rels).max() < 0.03, (
        f"{name} disagrees with the C++ energy albedo by {np.abs(rels).max():.2%} on cells "
        "that are not grazing and not extremely rough"
    )


@pytest.mark.parametrize("name,invert", [("ggx_glass_E", False), ("ggx_glass_inv_E", True)])
def test_both_glass_tables_run_high_at_grazing(name, invert):
    """LOCALITY. The bias is real, is one-sided, and is in BOTH tables -- so it is the
    1e3-sample generation, not a side/convention error in one of them. Asserted so that
    regenerating the tables trips this file rather than passing silently."""
    mi.set_variant(_variant())
    tab = np.array(_table(name).tensor())[..., 0]
    rels = []
    for k in Z_INDICES:
        eta_up = _reparam_eta(float(AXIS[k]))
        eta = 1.0 / eta_up if invert else eta_up
        for i, j in BIASED_CELLS:
            r, mu = float(AXIS[i]), float(AXIS[j])
            o = _oracle(r, mu, eta)
            rels.append((float(tab[k, j, i]) - o) / o)
    rels = np.asarray(rels)
    print(f"{name} grazing: {rels.size} cells, min {rels.min():+.2%}, max {rels.max():+.2%}")
    assert rels.size > 0
    assert rels.min() > 0.0, f"{name}'s grazing bias is no longer one-sided"
    assert rels.max() > 0.10, (
        f"{name}'s grazing bias has gone (max {rels.max():+.2%}) -- the table has been "
        "regenerated and the comment in `ggx_energy_compensation` is stale"
    )
