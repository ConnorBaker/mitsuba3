"""`blender_principled` interior-hit closure policy against Cycles.

WHAT THIS PINS. Cycles is double-sided by construction: `shader_setup_from_ray` flips
`sd->N` and `sd->Ng` on EVERY backfacing hit, so on an interior hit of a transmissive
material ALL closures shade on the flipped frame -- the glass closure takes exactly
`transmission_weight * weight` and the `1 - transmission` remainder flows to specular
and diffuse (`svm/closure.h`, `weight *= 1 - transmission_weight`, no side test). The
only side-aware quantity is the glass closure's IOR: `bsdf->ior = backfacing ?
1/ior : ior`. The port used to model the interior differently -- reflective lobes
side-masked off and the whole remaining budget re-routed to the transmission lobes --
which is a policy Cycles does not have, and it is the mechanism behind the measured
transmission-MIX residuals (pure transmission=0 and transmission=1 arms at parity while
the mix diverged). The port now mirrors unconditionally and carries the side in
`glass_eta` alone; these probes pin that policy.

CONTROLS, because each probe has a way of passing for free:
  * a transmission=1 arm where the diffuse budget must stay EXACTLY zero (the mirror
    must not ADMIT lobes Cycles doesn't give a budget to);
  * a front-side arm for every interior assertion (the instrument must distinguish
    the sides, not fire everywhere);
  * the MIS negative control from `test_valid_specular_reflection`, re-aimed at the
    ONE-SIDED interior path that the two-sided probe there cannot reach.
"""

import pytest

import drjit as dr
import mitsuba as mi


def _variant():
    for v in ("llvm_ad_rgb", "cuda_ad_rgb"):
        if v in mi.variants():
            return v
    pytest.skip("no suitable variant")


def _make_si(wi_local, sh_n=(0.0, 0.0, 1.0)):
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.p = mi.Point3f(0, 0, 0)
    si.n = mi.Vector3f(0, 0, 1)
    si.sh_frame = mi.Frame3f(dr.normalize(mi.Vector3f(*[float(v) for v in sh_n])))
    si.wi = dr.normalize(mi.Vector3f(*[float(v) for v in wi_local]))
    si.uv = mi.Point2f(0.5, 0.5)
    si.dp_du = mi.Vector3f(1, 0, 0)
    si.dp_dv = mi.Vector3f(0, 1, 0)
    si.wavelengths = mi.Color0f()
    return si


def _glassy(transmission, **kw):
    d = {"type": "blender_principled",
         "base_color": {"type": "rgb", "value": [0.8, 0.8, 0.8]},
         "roughness": 0.6, "metallic": 0.0,
         "transmission": float(transmission), "eta": 1.45}
    d.update(kw)
    return mi.load_dict(d)


def _sample_lanes(bsdf, si, n=256, seed=3):
    ctx = mi.BSDFContext()
    rng = mi.PCG32(size=n, initstate=mi.UInt64(seed))
    s1 = rng.next_float32()
    s2 = mi.Point2f(rng.next_float32(), rng.next_float32())
    return bsdf.sample(ctx, si, s1, s2, mi.Bool(True))


def test01_interior_remainder_reaches_the_diffuse_lobe():
    # THE POLICY PROBE. transmission = 0.5 leaves a 0.5 budget that Cycles hands to
    # specular+diffuse on BOTH sides of the surface. Ask eval for the diffuse lobe
    # alone: it must be alive on an interior hit -- and stay EXACTLY dead there when
    # transmission = 1, because then Cycles leaves it no budget either.
    mi.set_variant(_variant())

    ctx_diff = mi.BSDFContext()
    ctx_diff.type_mask = +mi.BSDFFlags.DiffuseReflection

    si_front = _make_si((0.4, 0.0, 0.917))
    si_back = _make_si((0.4, 0.0, -0.917))
    # outgoing on the same TRUE side as the view: a reflection configuration
    wo_front = dr.normalize(mi.Vector3f(-0.4, 0.0, 0.917))
    wo_back = dr.normalize(mi.Vector3f(-0.4, 0.0, -0.917))

    mix = _glassy(0.5)
    v_front = dr.max(mix.eval(ctx_diff, si_front, wo_front))[0]
    v_back = dr.max(mix.eval(ctx_diff, si_back, wo_back))[0]
    assert v_front > 0.0, "control vacuous: diffuse dead on the FRONT side too"
    assert v_back > 0.0, (
        "interior diffuse is dead -- the 1-transmission remainder is not carried "
        "on back-side hits (the old re-route-to-glass policy)")

    # CONTROL: at transmission = 1 there is no remainder, and the mirror must not
    # have invented one.
    pure = _glassy(1.0)
    assert dr.max(pure.eval(ctx_diff, si_back, wo_back))[0] == 0.0, (
        "transmission=1 interior hit grew a diffuse lobe -- the mirror is admitting "
        "budget Cycles does not grant")


def test02_interior_sampling_selects_reflective_lobes():
    # Same policy through the SAMPLING side: on an interior hit of the mix material
    # some lanes must pick diffuse or glossy REFLECTION (pre-policy: all lanes were
    # transmission-lobe lanes), and every surviving sample must keep wo on a
    # physical side (nonzero pdf, direction consistent with its component type).
    mi.set_variant(_variant())
    mix = _glassy(0.5)
    si_back = _make_si((0.4, 0.0, -0.917))

    bs, w = _sample_lanes(mix, si_back)
    alive = bs.pdf > 0.0
    assert int(dr.count(alive)[0]) >= 32, "probe vacuous: interior samples all dead"

    is_diffuse = alive & mi.has_flag(bs.sampled_type, mi.BSDFFlags.DiffuseReflection)
    assert int(dr.count(is_diffuse)[0]) > 0, (
        "no interior lane sampled the diffuse lobe -- the sampling weights still "
        "route the whole interior budget to transmission")

    # A refraction lane must EXIT the medium (true-space wo.z > 0 for wi.z < 0) and
    # carry the reciprocal relative IOR.
    is_refr = alive & mi.has_flag(bs.sampled_type, mi.BSDFFlags.GlossyTransmission)
    n_refr = int(dr.count(is_refr)[0])
    assert n_refr > 0, "probe vacuous: no refraction lanes survived"
    woz = dr.select(is_refr, bs.wo.z, 1.0)
    assert dr.min(woz)[0] > 0.0, "an interior refraction sample failed to exit"
    eta_err = dr.select(is_refr, dr.abs(bs.eta - 1.0 / 1.45), 0.0)
    assert dr.max(eta_err)[0] < 1e-6, "interior refraction eta is not 1/ior"

    # FRONT-SIDE CONTROL for the eta select: same material, same lanes, entering.
    bs_f, _ = _sample_lanes(mix, _make_si((0.4, 0.0, 0.917)))
    is_refr_f = (bs_f.pdf > 0.0) & mi.has_flag(bs_f.sampled_type,
                                               mi.BSDFFlags.GlossyTransmission)
    assert int(dr.count(is_refr_f)[0]) > 0
    eta_err_f = dr.select(is_refr_f, dr.abs(bs_f.eta - 1.45), 0.0)
    assert dr.max(eta_err_f)[0] < 1e-6, "front-side refraction eta is not ior"


def test03_interior_mis_agrees_and_the_control_fails():
    # sample().pdf must equal a fresh pdf() query on the ONE-SIDED interior path --
    # the path the two-sided probe in test_valid_specular_reflection cannot reach.
    # Smooth shading tilted off si.n so the correction frame is exercised too.
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs import blender_principled as BP

    mix = _glassy(0.5)
    si = _make_si((0.4, 0.1, -0.9), sh_n=(0.25, 0.05, 0.967))
    ctx = mi.BSDFContext()
    bs, _ = _sample_lanes(mix, si)
    pq = mix.pdf(ctx, si, bs.wo)
    act = bs.pdf > 0.0
    assert int(dr.count(act)[0]) >= 16, "probe vacuous: samples all rejected"
    rel = dr.select(act, dr.abs(bs.pdf - pq) / dr.maximum(bs.pdf, 1e-9), 0.0)
    assert dr.max(rel)[0] < 1e-4, "interior MIS: sample and query pdf disagree"

    # NEGATIVE CONTROL: drop the pre-mirror sign; the same check must fail, or this
    # probe could not see the bug it guards.
    real = BP.BlenderPrincipledBSDF._eval_pdf_impl

    def broken(self, attr, ctx, si, wo_, active, is_eval=True, wiz0=None):
        return real(self, attr, ctx, si, wo_, active, is_eval=is_eval, wiz0=None)

    BP.BlenderPrincipledBSDF._eval_pdf_impl = broken
    try:
        bs2, _ = _sample_lanes(mix, si)
    finally:
        BP.BlenderPrincipledBSDF._eval_pdf_impl = real
    pq2 = mix.pdf(ctx, si, bs2.wo)
    rel2 = dr.select(bs2.pdf > 0.0,
                     dr.abs(bs2.pdf - pq2) / dr.maximum(bs2.pdf, 1e-9), 0.0)
    assert dr.max(rel2)[0] > 1e-3, (
        "negative control failed -- the interior MIS check cannot see the wiz0 bug")
