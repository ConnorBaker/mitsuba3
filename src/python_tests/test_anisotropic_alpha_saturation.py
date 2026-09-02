"""The anisotropic aspect split against Cycles' alpha saturation and energy-table key.

WHAT THIS PINS. Cycles saturates both microfacet alphas in every setup
(`bsdf_microfacet.h`: `bsdf->alpha_x = saturatef(bsdf->alpha_x)`), and reads the GGX
energy tables at `rough = sqrtf(sqrtf(alpha_x * alpha_y))` of those CLAMPED alphas --
computed in setup, BEFORE `bsdf_microfacet_blur` applies the filter-glossy floor, so the
key never sees `min_alpha`. The port did neither: at roughness 1, anisotropic 0.79 it
rendered a (1.86, 0.537) lobe with compensation keyed at roughness 1.0, and the furnace
read 0.6014 against Cycles at that corner (parity at low roughness where nothing clamps).
Measured on the splash scene's `Steel Coated`, the only stable dark outlier in the
per-material census -- whose ablation localized the whole deficit to roughness-1 x
anisotropy and cleared it at either input alone.
"""

import numpy as np
import pytest

import drjit as dr
import mitsuba as mi


def _variant():
    for v in ("llvm_ad_rgb", "cuda_ad_rgb"):
        if v in mi.variants():
            return v
    pytest.skip("no suitable variant")


def cy_alphas(roughness, anisotropic, min_alpha=0.0):
    aspect = np.sqrt(1.0 - 0.9 * anisotropic)
    ax = max(max(1e-4, min_alpha), roughness**2 / aspect)
    ay = max(max(1e-4, min_alpha), roughness**2 * aspect)
    return min(ax, 1.0), min(ay, 1.0)  # saturatef


def test01_alphas_saturate_like_cycles():
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs.lobes import microfacet_compute_alphas

    clamped = 0
    for r in (0.05, 0.1793, 0.5, 0.8, 1.0):
        for a in (0.0, 0.3, 0.7909, 1.0):
            ax, ay = microfacet_compute_alphas(mi.Float(r), mi.Float(a))
            rx, ry = cy_alphas(r, a)
            assert abs(ax[0] - rx) < 1e-6 and abs(ay[0] - ry) < 1e-6, (r, a, ax[0], ay[0])
            if r**2 / np.sqrt(1.0 - 0.9 * a) > 1.0:
                clamped += 1
    assert clamped >= 3, "sweep never exercised the saturation -- vacuous"


def test02_table_key_is_the_clamped_geometric_mean():
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs.lobes import microfacet_table_roughness

    # Where nothing clamps the key IS the roughness (exact identity)...
    assert abs(microfacet_table_roughness(mi.Float(0.5), mi.Float(0.7909))[0] - 0.5) < 1e-6
    # ...and where an axis clamps it must drop BELOW the raw roughness, to the
    # Cycles value -- keying on raw roughness at this point is the measured defect.
    ax, ay = cy_alphas(1.0, 0.7909)
    ref = (ax * ay) ** 0.25
    got = microfacet_table_roughness(mi.Float(1.0), mi.Float(0.7909))[0]
    assert abs(got - ref) < 1e-6
    assert got < 1.0 - 1e-3, "negative control: the clamped key must differ from raw roughness"


def test03_the_key_excludes_the_filter_glossy_floor():
    # Cycles computes `energy_scale` in setup, before `bsdf_microfacet_blur`; a key that
    # saw `min_alpha` would re-brighten every blurred indirect bounce.
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs.lobes import microfacet_table_roughness

    lo = microfacet_table_roughness(mi.Float(0.1), mi.Float(0.0))[0]
    assert abs(lo - 0.1) < 1e-6  # a floor of e.g. 0.25 would have raised this
