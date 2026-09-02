"""The `blender_lamp` emitter owns Cycles' directional lamp attenuation -- and both of its
sampling strategies must see the SAME attenuation.

It replaces the previous composition (`area` + the `blender_light_attenuation` radiance
texture), which carried the emitted direction to the attenuation THROUGH the texture's
`si.wi` -- workable only because `area.cpp` was patched to populate `si.wi` at every
texture-evaluation site, and reliant on the texture-importance-sampling branch for SPOT.
Here the emitter applies the attenuation from geometry it holds in every code path
(`si.wi` on the hit path, `-ds.d` in both direct-sampling halves, the sampled direction in
`sample_ray`), and both modes use the attached shape's solid-angle sampling, so
MIS-consistency is structural rather than patched in.

Every expected value below is computed IN THE TEST from Cycles' own formulas
(`spot_light_attenuation`, intern/cycles/kernel/light/spot.h;
`area_light_spread_attenuation`, intern/cycles/kernel/light/area.h) -- an independent
transcription, so the emitter is not checked against itself. At the prototype stage the
plugin was also rendered against the legacy composition in a closed two-wall box on
cuda_ad_rgb: AREA_SPREAD agreed bit-for-bit (identical estimator once both sides use shape
sampling); SPOT agreed to 0.1% on the image mean at 512 spp under a deliberately different
sampling strategy (solid-angle here vs the texture-importance branch there).
"""

import math

import pytest
import drjit as dr
import mitsuba as mi


# 60-degree cone, blend 0.4 (Cycles: spot_smooth = 1 / ((1 - cos_half) * blend))
COS_HALF = math.cos(math.radians(30.0))
BLEND = 0.4
SPOT_SMOOTH = 1.0 / ((1.0 - COS_HALF) * BLEND)

# 45-degree spread (Cycles: normalize_spread = 1 / (tan(h) - h))
HALF_SPREAD = math.radians(22.5)
TAN_HALF_SPREAD = math.tan(HALF_SPREAD)
NORM_SPREAD = 1.0 / (TAN_HALF_SPREAD - HALF_SPREAD)

RAD = 2.5


def spot_atten_ref(d_world):
    """`spot_light_attenuation` for an identity lamp frame (+z the cone axis)."""
    n = math.sqrt(sum(c * c for c in d_world))
    z = d_world[2] / n
    t = min(max((z - COS_HALF) * SPOT_SMOOTH, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def spread_atten_ref(cos_local):
    """`area_light_spread_attenuation` for a flat lamp."""
    s = math.sqrt(max(1.0 - cos_local * cos_local, 0.0))
    tan_a = s / max(cos_local, 1e-30)
    return max((TAN_HALF_SPREAD - tan_a) * NORM_SPREAD, 0.0)


def _receiver(p):
    it = dr.zeros(mi.Interaction3f)
    it.p = mi.Point3f(*p)
    return it


def _spot_emitter():
    """Returns (shape, emitter). The SHAPE must stay referenced for as long as the
    emitter is used: `Emitter::m_shape` is a raw C++ back-pointer, so dropping the shape
    (`return shape.emitter()`) leaves the emitter pointing at freed memory -- measured
    here as a segfault in `sample_direction` before this helper kept it alive."""
    shape = mi.load_dict({
        'type': 'sphere', 'radius': 0.1,
        'emitter': {'type': 'blender_lamp', 'mode': 'SPOT',
                    'radiance': {'type': 'rgb', 'value': RAD},
                    'cos_half_spot_angle': COS_HALF, 'spot_smooth': SPOT_SMOOTH},
    })
    return shape, shape.emitter()


def _spread_emitter(tan_half_spread=TAN_HALF_SPREAD, normalize_spread=NORM_SPREAD):
    """Returns (shape, emitter) -- see `_spot_emitter` for why the shape rides along."""
    shape = mi.load_dict({
        'type': 'rectangle',
        'emitter': {'type': 'blender_lamp', 'mode': 'AREA_SPREAD',
                    'radiance': {'type': 'rgb', 'value': RAD},
                    'tan_half_spread': tan_half_spread,
                    'normalize_spread': normalize_spread},
    })
    return shape, shape.emitter()


def _first(x):
    return float(dr.slice(dr.mean(x, axis=None) if dr.depth_v(x) > 1 else x)
                 if dr.width(x) else 0.0)


@pytest.mark.parametrize('p', [(0.02, -0.01, 3.0),    # inside the cone
                               (1.55, 0.0, 3.0),      # on the smoothstep ramp
                               (3.0, 0.0, 1.0)])      # outside the cone (atten == 0)
def test01_spot_sample_direction_matches_cycles(variants_vec_backends_once_rgb, p):
    """`sample_direction`'s radiance (weight * pdf) equals Cycles' cone attenuation,
    computed here from `ds` -- geometry the caller already holds."""
    sh, em = _spot_emitter()
    it = _receiver(p)
    ds, w = em.sample_direction(it, mi.Point2f(0.4, 0.6), True)
    got = _first(w * ds.pdf)
    d = [-float(dr.slice(ds.d[i])) for i in range(3)]
    expect = RAD * spot_atten_ref(d)
    assert abs(got - expect) < 1e-4 * max(expect, 1.0), (got, expect)


@pytest.mark.parametrize('p', [(0.02, -0.01, 3.0), (1.55, 0.0, 3.0)])
def test02_spot_mis_consistency(variants_vec_backends_once_rgb, p):
    """The two direct-illumination strategies must agree: eval_direction == weight * pdf,
    and pdf_direction == the pdf the sample was drawn from."""
    sh, em = _spot_emitter()
    it = _receiver(p)
    ds, w = em.sample_direction(it, mi.Point2f(0.4, 0.6), True)
    from_sample = _first(w * ds.pdf)
    from_eval = _first(em.eval_direction(it, ds, True))
    assert abs(from_eval - from_sample) < 1e-5 * max(from_sample, 1.0) + 1e-7
    pdf = float(dr.slice(em.pdf_direction(it, ds, True)))
    ds_pdf = float(dr.slice(ds.pdf))
    assert abs(pdf - ds_pdf) < 1e-6 * max(ds_pdf, 1.0)


@pytest.mark.parametrize('p', [(0.05, -0.1, 4.0),     # near-axis (atten > 0)
                               (1.2, 0.4, 2.0),       # oblique (outside the spread)
                               (4.0, 0.0, 1.0)])      # far outside
def test03_spread_sample_direction_matches_cycles(variants_vec_backends_once_rgb, p):
    sh, em = _spread_emitter()
    it = _receiver(p)
    ds, w = em.sample_direction(it, mi.Point2f(0.3, 0.7), True)
    got = _first(w * ds.pdf)
    cos_local = -sum(float(dr.slice(ds.d[i])) * float(dr.slice(ds.n[i]))
                     for i in range(3))
    expect = RAD * spread_atten_ref(cos_local)
    assert abs(got - expect) < 1e-4 * max(expect, 1.0), (got, expect)
    from_eval = _first(em.eval_direction(it, ds, True))
    assert abs(from_eval - got) < 1e-5 * max(got, 1.0) + 1e-7


def test04_the_check_can_fail(variants_vec_backends_once_rgb):
    """VACUITY CONTROL: a receiver behind the one-sided lamp gets exactly zero -- so the
    tests above are reading a real quantity, not a constant that happens to be positive."""
    sh, em = _spread_emitter()
    it = _receiver((0.0, 0.0, -3.0))
    ds, w = em.sample_direction(it, mi.Point2f(0.3, 0.7), True)
    assert _first(w) == 0.0


def test05_hit_path_agrees_with_the_same_formula(variants_vec_backends_once_rgb):
    """`eval` -- the strategy fed by BSDF sampling -- must see the same attenuation as the
    NEE half. This is the cross-strategy agreement whose absence was the original
    spread-lamp blackness (ratio of means 0.0008 vs Cycles)."""
    scene = mi.load_dict({
        'type': 'scene',
        'lamp': {'type': 'rectangle',
                 'emitter': {'type': 'blender_lamp', 'mode': 'AREA_SPREAD',
                             'radiance': {'type': 'rgb', 'value': RAD},
                             'tan_half_spread': TAN_HALF_SPREAD,
                             'normalize_spread': NORM_SPREAD}},
    })
    ray = mi.Ray3f(mi.Point3f(0.3, -0.2, 3.0),
                   dr.normalize(mi.Vector3f(-0.12, 0.1, -1.0)))
    si = scene.ray_intersect(ray)
    val = _first(si.emitter(scene).eval(si, True))
    cos_local = float(dr.slice(mi.Frame3f.cos_theta(si.wi)))
    expect = RAD * spread_atten_ref(cos_local)
    assert abs(val - expect) < 1e-4 * max(expect, 1.0), (val, expect)


def test06_collimated_arm(variants_vec_backends_once_rgb):
    """`tan_half_spread == 0`: dark off-axis, pi on-axis (Cycles' fully-collimated arm)."""
    sh, em = _spread_emitter(tan_half_spread=0.0, normalize_spread=0.0)
    on_axis = _receiver((0.0, 0.0, 5.0))
    # The shape samples positions across the rectangle, so a receiver on the axis still
    # sees off-axis directions almost surely; sample the center.
    ds, w = em.sample_direction(on_axis, mi.Point2f(0.5, 0.5), True)
    got = _first(w * ds.pdf)
    assert abs(got - RAD * math.pi) < 1e-3 * RAD * math.pi, got
    off = _receiver((3.0, 0.0, 0.5))
    ds, w = em.sample_direction(off, mi.Point2f(0.3, 0.7), True)
    assert _first(w) == 0.0


def test07_unknown_mode_is_refused(variants_vec_backends_once_rgb):
    with pytest.raises(Exception, match='mode'):
        mi.load_dict({'type': 'rectangle',
                      'emitter': {'type': 'blender_lamp', 'mode': 'CONE',
                                  'radiance': {'type': 'rgb', 'value': 1.0}}})
