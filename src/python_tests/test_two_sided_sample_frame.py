"""`blender_principled.sample()` must return `wo` in the CALLER's frame, not the flipped one.

For `two_sided` materials `_sample_impl` mirrors `si.wi` into the upper hemisphere, samples
there, and is supposed to mirror the sampled direction back on the way out. It did not: the
flip-back read `si.wi.z`, which the mirror step had already overwritten with `dr.abs(...)`,
so `dr.mulsign` could only ever multiply by +1. `_eval_pdf_impl` performs the same two steps
in the opposite order and was therefore correct -- which is what makes the bug invisible to
any test that only checks `eval`/`pdf`.

The consequence is not subtle. Every back-face BSDF sample is handed to the integrator
pointing INTO the surface, so the ray hits nothing and the BSDF-sampling half of MIS
contributes zero. Because MIS still discounts the emitter-sampling half against a healthy
`bsdf_pdf`, the result is a NET LOSS rather than merely a missed strategy: on an isolated
back-lit test object the `direct` integrator scored 5.8x LOWER with MIS than with emitter
sampling alone, and the whole object rendered at 0.13x of Cycles.

Two independent assertions, because either alone can be satisfied by a broken sampler:
  1. the returned `wo` is in the same hemisphere as `wi` (a reflective, opaque material
     cannot scatter through itself), and
  2. `pdf(wo)` agrees with the `pdf` reported alongside it -- the canonical sample/pdf
     consistency check, which is what a wrongly-framed direction violates.
"""

import pytest
import drjit as dr
import mitsuba as mi


def make_bsdf():
    return mi.load_dict({
        'type': 'blender_principled',
        'base_color': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]},
        'roughness': 0.5,
        'metallic': 0.0,
        'transmission': 0.0,
        'alpha': 1.0,
        'two_sided': True,
    })


def make_si(n, wi_z_sign):
    si = dr.zeros(mi.SurfaceInteraction3f, n)
    si.p = mi.Point3f(0.0, 0.0, 0.0)
    si.n = mi.Normal3f(0.0, 0.0, 1.0)
    si.sh_frame = mi.Frame3f(si.n)
    si.wi = dr.normalize(mi.Vector3f(0.3, 0.2, 0.93 * wi_z_sign))
    # A degenerate tangent frame produces NaNs that mask the effect under test.
    si.dp_du = mi.Vector3f(1.0, 0.0, 0.0)
    si.dp_dv = mi.Vector3f(0.0, 1.0, 0.0)
    si.uv = mi.Point2f(0.5, 0.5)
    si.t = mi.Float(1.0)
    return si


@pytest.mark.parametrize('wi_z_sign', [1.0, -1.0], ids=['front_face', 'back_face'])
def test_two_sided_sample_returns_wo_in_caller_frame(variants_vec_backends_once_rgb,
                                                     wi_z_sign):
    n = 2048
    b = make_bsdf()
    si = make_si(n, wi_z_sign)
    ctx = mi.BSDFContext()

    rng = mi.PCG32(size=n)
    s1 = rng.next_float32()
    s2 = mi.Point2f(rng.next_float32(), rng.next_float32())
    bs, weight = b.sample(ctx, si, s1, s2, True)

    live = dr.all(dr.isfinite(weight), axis=None) and True
    assert live, 'sample() returned non-finite weights'

    active = (bs.pdf > 0.0) & (dr.max(weight, axis=0) > 0.0)
    n_active = dr.count(active)[0]
    # VACUITY: an all-dead mask satisfies every assertion below for free.
    assert n_active > n // 4, \
        f'only {n_active} of {n} samples were live; the probe proves nothing'

    # (1) An opaque, non-transmissive material must not scatter through itself.
    same_side = (si.wi.z * bs.wo.z) > 0.0
    n_through = dr.count(active & ~same_side)[0]
    assert n_through == 0, (
        f'{n_through} of {n_active} sampled directions are on the opposite side of the '
        f'surface from wi on a material with transmission=0, alpha=1 -- sample() is '
        f'returning wo in the internally flipped frame')

    # (2) Canonical sample/pdf consistency at the direction sample() just returned.
    pdf_at = b.pdf(ctx, si, bs.wo, active)
    agree = dr.abs(pdf_at - bs.pdf) <= 1e-2 * dr.maximum(bs.pdf, 1e-12)
    n_bad = dr.count(active & ~agree)[0]
    assert n_bad == 0, (
        f'pdf(bs.wo) disagrees with bs.pdf on {n_bad} of {n_active} samples')
