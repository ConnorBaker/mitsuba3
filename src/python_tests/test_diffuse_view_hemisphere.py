"""Cycles' diffuse-range closures do not test the VIEW against the closure normal.

WHAT THIS PINS. When a bump or normal map tilts a closure normal `N'` far enough, the view
direction can fall below `N'`'s horizon while the surface is still perfectly front-facing
geometrically. Cycles shades those points normally:

  * `bsdf_diffuse_eval(sc, const float3 /*wi*/, wo, pdf)` -- the incident direction is
    declared unnamed and unused -- returns `max(dot(N', wo), 0) * M_1_PI`;
  * `bsdf_sheen_eval` does the same;
  * `bsdf_oren_nayar_eval` gates on `cosNO > 0` and `bsdf_oren_nayar_get_intensity` CLAMPS
    the view cosine (`nv = max(dot(n, v), 0)`) rather than rejecting on it;
  * `bsdf_eval` applies `bump_shadowing_term(sd, sc, wo, true)` to the OUTGOING direction.

Nothing in that path tests the view. Only the MICROFACET closures do, in
`bsdf_microfacet_eval`, which rejects `cos_NI <= 0` -- so the two-direction test is correct
there and wrong for the diffuse range.

WHY IT MATTERS, MEASURED. On a plane carrying a sinusoidal height map at peak height gradient
201/uv, 43.5% of hit pixels had the view below the bumped horizon and `blender_principled`
returned zero sample weight, zero pdf and zero eval for every one of them -- black in Mitsuba,
lit in Cycles. Whole-image mi/cy at that operating point was 0.6337; with the view test
removed from the diffuse-range lobes it is 1.0024, and the 16-cell frequency x amplitude grid
that surrounds it is within 0.33% everywhere.

WHERE THE BUG ACTUALLY WAS, because it is not where it looks. `_eval_pdf_impl` has per-lobe
masks and those were relaxed FIRST; the panel came back BIT-IDENTICAL. `_lobe_weights` had
already zeroed `weights.diffuse` and `weights.sheen` through its own `&= front_side`, so the
relaxed mask had nothing left to admit. Both edits are required and neither moves a pixel
alone. This test exercises the composite through the public `eval` / `sample`, so it cannot be
satisfied by fixing only one of them.

THE PROPERTY, chosen so no constant has to be trusted. Configure the plugin as a pure
Lambertian -- `diffuse_roughness = 0` takes `OrenNayarLobe`'s `B <= 0` branch, which is
`cos_theta_o / pi` -- and turn the bump correction off so the GGX softening is out of the way.
Cycles' value then depends on `wo` alone, so `eval` at a view BELOW the closure horizon must
EQUAL `eval` at a view above it, for the same `wo`. That is an identity, not a fitted number.

THREE CONTROLS, because "returns something nonzero" passes for free:
  * VACUITY -- the below-horizon view must really be below it, and enough outgoing directions
    must evaluate strictly positive, or the identity is 0 == 0;
  * SHAPE -- `eval / dot(N', wo)` must be constant across `wo`, which pins the Lambertian
    form rather than merely its liveness;
  * NOT-OVER-RELAXED -- a metallic (microfacet) lobe must STILL be rejected below the
    horizon. Removing the view test from every lobe would pass every other assertion here.
"""

import numpy as np
import pytest

import drjit as dr
import mitsuba as mi


def _variant():
    for v in ("llvm_ad_rgb", "cuda_ad_rgb", "scalar_rgb"):
        if v in mi.variants():
            return v
    pytest.skip("no suitable variant")


def _tol():
    return 1e-9 if dr.type_v(mi.Float) == dr.VarType.Float64 else 1e-6


def _v3(a):
    # mi.Vector3f rejects np.float64 elements.
    return mi.Vector3f(*[float(x) for x in a])


# A steeply tilted tangent-space normal: 80 deg off the geometric normal, in +x.
TILT = np.radians(80.0)
NC = np.array([np.sin(TILT), 0.0, np.cos(TILT)])

# Two views, both on the geometric FRONT side (z > 0), straddling N''s horizon.
WI_ABOVE = np.array([0.5, 0.0, np.sqrt(3) / 2])          # dot(N', wi) = +0.6427
WI_BELOW = np.array([-0.5, 0.0, np.sqrt(3) / 2])         # dot(N', wi) = -0.8368


def _bsdf(metallic=0.0, roughness=1.0):
    return mi.load_dict({
        'type': 'blender_principled',
        'two_sided': True, 'multiscatter': False,
        'base_color': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]},
        'roughness': roughness, 'metallic': metallic, 'eta': 1.45, 'alpha': 1.0,
        # 0 takes OrenNayarLobe's `B <= 0` Lambertian branch, so the reference value is
        # `cos_theta_o / pi` and carries no view dependence of its own.
        'diffuse_roughness': 0.0,
        'spec_ior_level': 0.0, 'spec_tint': {'type': 'rgb', 'value': [1, 1, 1]},
        'anisotropic': 0.0, 'anisotropic_rot': 0.0, 'transmission': 0.0,
        'clearcoat': 0.0, 'clearcoat_roughness': 0.03, 'clearcoat_ior': 1.5,
        'clearcoat_tint': {'type': 'rgb', 'value': [1, 1, 1]},
        'sheen': 0.0, 'sheen_roughness': 0.5,
        'sheen_tint': {'type': 'rgb', 'value': [1, 1, 1]},
        'normalmap': {'type': 'rgb',
                      'value': [0.5 + 0.5 * float(NC[0]),
                                0.5 + 0.5 * float(NC[1]),
                                0.5 + 0.5 * float(NC[2])]},
        # OFF so the GGX softening -- which is a function of the OUTGOING direction and would
        # scale both arms identically anyway -- is not part of what is being asserted.
        'bump_map_correction': False,
    })


def _si(wi, n=1):
    """A COMPLETE interaction. `si.sh_frame.n = ...` alone is not one.

    `dr.zeros` leaves `sh_frame.s` and `sh_frame.t` at (0, 0, 0), and assigning only `.n`
    does not fill them -- the frame stays degenerate. `eval` happens to survive that (it
    works in the closure frame, which is rebuilt from `dp_du`/`dp_dv`), so the defect is
    invisible in an eval-only test; `sample` does not, and returns pdf 0 and a zero
    direction for EVERY lane, above and below the horizon alike. Building this file that
    way produced a "sampling is dead below the closure horizon" result and a `nan` from the
    metallic control, both of which read exactly like plugin bugs and were neither.
    """
    si = dr.zeros(mi.SurfaceInteraction3f, n)
    si.p = mi.Point3f(0, 0, 0)
    si.n = mi.Normal3f(0, 0, 1)
    si.sh_frame = mi.Frame3f(mi.Normal3f(0, 0, 1))
    si.dp_du, si.dp_dv = mi.Vector3f(1, 0, 0), mi.Vector3f(0, 1, 0)
    si.wi = _v3(wi)
    return si


# Outgoing directions in the +x half, all above the geometric plane and above N''s horizon.
WOS = [np.array([np.sin(a), 0.0, np.cos(a)]) for a in np.radians([5, 20, 40, 60, 75, 85])]


def _evals(bsdf, wi):
    ctx = mi.BSDFContext()
    si = _si(wi)
    return [float(mi.luminance(bsdf.eval(ctx, si, _v3(wo)))[0]) for wo in WOS]


def test01_the_probe_is_not_vacuous():
    """The below-horizon view must be below it, and the identity must not be 0 == 0."""
    mi.set_variant(_variant())
    assert float(np.dot(NC, WI_BELOW)) < 0.0, (
        "WI_BELOW is not actually below the closure horizon (dot = %.4f); the whole test "
        "would then compare two above-horizon views" % float(np.dot(NC, WI_BELOW)))
    assert float(np.dot(NC, WI_ABOVE)) > 0.0
    assert WI_BELOW[2] > 0.0 and WI_ABOVE[2] > 0.0, (
        "both views must be on the geometric front side, or a geometric back-face rejection "
        "is what is being measured")

    above = _evals(_bsdf(), WI_ABOVE)
    positive = sum(v > 1e-6 for v in above)
    assert positive >= 4, (
        "only %d of %d outgoing directions evaluate positive even with the view ABOVE the "
        "horizon -- the identity in test02 would be satisfied by zeros"
        % (positive, len(WOS)))


def test02_diffuse_ignores_the_view_hemisphere():
    """Cycles' `bsdf_diffuse_eval` takes `const float3 /*wi*/`. So must this."""
    mi.set_variant(_variant())
    bsdf = _bsdf()
    above = _evals(bsdf, WI_ABOVE)
    below = _evals(bsdf, WI_BELOW)
    tol = _tol()
    worst = max(abs(a - b) for a, b in zip(above, below))
    assert worst < tol, (
        "eval depends on which side of the closure horizon the VIEW is on: worst "
        "|above - below| = %.3e over %d outgoing directions, tolerance %.1e.\n"
        "  above: %s\n  below: %s\n"
        "A row of zeros in `below` is the pre-fix behaviour -- `_lobe_weights` masking "
        "`masks.diffuse &= front_side` on the SHADING frame."
        % (worst, len(WOS), tol,
           " ".join("%.6f" % v for v in above), " ".join("%.6f" % v for v in below)))


def test03_the_value_is_lambertian_in_the_outgoing_direction():
    """SHAPE: liveness is not enough -- the value must be `dot(N', wo)`-proportional."""
    mi.set_variant(_variant())
    below = _evals(_bsdf(), WI_BELOW)
    ratios = [v / float(np.dot(NC, wo)) for v, wo in zip(below, WOS)
              if float(np.dot(NC, wo)) > 1e-3]
    assert len(ratios) >= 4
    spread = max(ratios) - min(ratios)
    assert spread < 1e-4 * max(ratios), (
        "eval / dot(N', wo) is not constant across the outgoing sweep (spread %.3e on a "
        "mean of %.6f), so the lobe is alive below the horizon but is not the Lambertian "
        "`max(dot(N', wo), 0) / pi` Cycles evaluates there: %s"
        % (spread, float(np.mean(ratios)), " ".join("%.6f" % r for r in ratios)))


def test04_sampling_is_alive_and_agrees_with_eval():
    """A live `eval` with a dead `sample` is still a black pixel in a path tracer."""
    mi.set_variant(_variant())
    bsdf = _bsdf()
    ctx = mi.BSDFContext()
    n = 64
    rng = np.random.default_rng(0)
    si = _si(WI_BELOW, n)

    s1 = mi.Float(rng.random(n).astype(np.float32))
    s2 = mi.Point2f(rng.random(n).astype(np.float32), rng.random(n).astype(np.float32))
    bs, weight = bsdf.sample(ctx, si, s1, s2, mi.Bool(True))
    pdf = np.array(bs.pdf)
    live = pdf > 0.0
    assert live.mean() > 0.4, (
        "only %.1f%% of samples taken with the view BELOW the closure horizon have a "
        "positive pdf; before the fix this was 0%%" % (100.0 * live.mean()))

    # sample/eval consistency: the pdf reported by `sample` must be the pdf `eval_pdf`
    # reports for the same direction, or MIS weights the contribution wrongly.
    _, pdf2 = bsdf.eval_pdf(ctx, si, bs.wo, mi.Bool(True))
    pdf2 = np.array(pdf2)
    rel = np.abs(pdf2[live] - pdf[live]) / np.maximum(pdf[live], 1e-9)
    assert rel.max() < 1e-4, (
        "sample() and eval_pdf() disagree about the pdf below the horizon by up to %.3e "
        "relative" % rel.max())

    w = np.array(mi.luminance(weight))[live] if hasattr(mi, 'luminance') \
        else np.array(weight[0])[live]
    assert np.isfinite(w).all() and (w >= 0).all()


def test05_microfacet_lobes_still_reject_the_view():
    """NOT-OVER-RELAXED: `bsdf_microfacet_eval` DOES reject `cos_NI <= 0`."""
    mi.set_variant(_variant())
    metal = _bsdf(metallic=1.0, roughness=0.3)
    above = _evals(metal, WI_ABOVE)
    below = _evals(metal, WI_BELOW)
    assert np.isfinite(above).all() and np.isfinite(below).all(), (
        "the metallic control produced a non-finite value (above: %s, below: %s). A `nan` "
        "here was, once, a degenerate `si.sh_frame` in this file rather than a plugin "
        "defect -- see `_si`." % (above, below))
    assert max(above) > 1e-6, (
        "the metallic control evaluates to zero even ABOVE the horizon (max %.3e), so it "
        "cannot detect an over-relaxed view test" % max(above))
    assert max(below) <= 1e-9, (
        "a microfacet lobe evaluates below the closure horizon (max %.3e). Cycles rejects "
        "`cos_NI <= 0` there; the view test was removed from too many lobes." % max(below))


# --- intern/cycles/kernel/closure/bsdf_oren_nayar.h ------------------------------------
def _cy_G(c):
    if c < 1e-6:
        # "The tan(theta) term starts to act up at low cosTheta, so fall back to Taylor."
        return (np.pi / 2 - 2.0 / 3.0) - c
    s_ = np.sqrt(max(1.0 - c * c, 0.0))
    return (s_ * (np.arccos(c) - 2.0 / 3.0 - s_ * c)
            + 2.0 / 3.0 * (s_ / c) * (1.0 - s_ ** 3))


def cy_oren_nayar(n, v, l, albedo, roughness):
    """`bsdf_oren_nayar_get_intensity` composed with `bsdf_oren_nayar_param`.

    `param` is built once at closure SETUP from `dot(bsdf->N, sd->wi)` -- the view -- and
    `get_intensity` supplies the outgoing direction per evaluation, so the multiscatter
    factor is `Ems * (1 - Ev(view)) * (1 - El(outgoing))`. Both `nv` reads are clamped at
    the source: `max(dot(n, v), 0)` in `get_intensity` and `max(nv, 0)` in `param`.
    """
    sigma = min(max(roughness, 0.0), 1.0)
    a = 1.0 / (np.pi + sigma * (np.pi / 2 - 2.0 / 3.0))
    b = sigma * a
    nl = max(float(np.dot(n, l)), 0.0)
    if b <= 0.0:
        return nl / np.pi
    nv = max(float(np.dot(n, v)), 0.0)
    Eavg = a * np.pi + ((2 * np.pi - 5.6) / 3.0) * b
    Ems = (1.0 / np.pi) * albedo ** 2 * (Eavg / (1.0 - Eavg)) / (1.0 - albedo * (1.0 - Eavg))
    ms_term = Ems * (1.0 - (a * np.pi + b * _cy_G(nv)))
    t = float(np.dot(l, v)) - nl * nv
    if t > 0.0:
        t /= max(nl, nv) + 1e-8          # Cycles uses FLT_MIN; see the tolerance note below
    single = a + b * t
    multi = ms_term * (1.0 - (a * np.pi + b * _cy_G(nl)))
    return nl * (single + multi)


def test06_oren_nayar_matches_the_kernel_below_the_horizon():
    """The clamp in `OrenNayarLobe.eval_pdf` is only REACHABLE after this fix.

    `bsdf_oren_nayar_get_intensity` computes `nl = max(dot(n, l), 0)` and
    `nv = max(dot(n, v), 0)`; `OrenNayarLobe` matches that with two `dr.maximum(..., 0)`.
    While the caller masked the lobe off for any view below the closure horizon the clamp
    could never fire, so removing it was a no-op and nothing would have noticed.

    IT IS NOT ENOUGH TO ASSERT THAT THE RESULT IS FINITE. That was the first version of this
    test and it did not discriminate: with the clamp removed, `oren_nayar_G(-0.342)` is
    perfectly finite -- `acos` and `tan` both accept it -- and the value stays positive. The
    unclamped lobe is WRONG, not broken, so only a comparison against the kernel catches it.
    Verified by mutation: deleting both `dr.maximum` calls makes this test fail and left the
    finite/positive version passing.

    Tolerance: the transcription uses `1e-8` where the kernel uses `FLT_MIN` in the `t`
    denominator. That differs only when `max(nl, nv)` is itself near zero, which the sweep
    avoids by keeping every outgoing direction well above the closure normal.
    """
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs.lobes import OrenNayarLobe

    N = np.array([0.0, 0.0, 1.0])        # the lobe works in the closure frame
    ALBEDO, ROUGH = 0.8, 0.9
    views = [np.array([np.sin(t), 0.0, np.cos(t)]) for t in
             np.radians([10, 40, 70, 95, 110, 130, 160])]
    outs = [np.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)])
            for t in np.radians([10, 30, 55, 75]) for ph in np.radians([0, 90, 180])]

    worst, n_below, n_cases = 0.0, 0, 0
    for v in views:
        for l in outs:
            val = OrenNayarLobe.eval_pdf(
                _v3(v), _v3(l), mi.UnpolarizedSpectrum(ALBEDO), mi.Float(ROUGH))[0]
            got = float(np.array(val).ravel()[0])
            exp = cy_oren_nayar(N, v, l, ALBEDO, ROUGH)
            worst = max(worst, abs(got - exp))
            n_below += v[2] < 0.0
            n_cases += 1

    assert n_below > 0, (
        "no case in the sweep has the view below the closure normal, so the clamp under "
        "test is never reached")
    assert worst < 3e-5, (
        "OrenNayarLobe disagrees with `bsdf_oren_nayar_get_intensity` by %.3e over %d cases "
        "(%d of them with the view below the closure normal). An unclamped `cos_theta_i` "
        "reaches `oren_nayar_G` and the `t` denominator." % (worst, n_cases, n_below))
