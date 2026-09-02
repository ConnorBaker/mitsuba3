"""`blender_principled` against Cycles' bump-map correction (`bump_shadowing_term`).

WHAT THIS PINS. Cycles applies a per-material correction, defaulted ON
(`SOCKET_BOOLEAN(use_bump_map_correction, "Bump Map Correction", true)` in
`intern/cycles/scene/shader.cpp`), whenever a closure's normal differs from the smooth
shading normal. It has two halves with DIFFERENT scope, and conflating them is the easy
mistake:

  * a hard hemisphere-consistency rejection, applied to every closure on eval and to
    diffuse-range ones only when sampling, and NOT gated by the material flag; and
  * a GGX shadowing-masking factor applied to diffuse-range closures only, which IS gated.

`blender_principled` implemented neither. Measured on a plane with a constant tilted normal
map under a single delta sun, Mitsuba reproduced Cycles-with-the-correction-DISABLED to
seven digits and missed Cycles-as-shipped by up to 5.59x at grazing incidence.

WHY A TRANSCRIPTION RATHER THAN A GOLDEN IMAGE. The kernel function is 40 lines of pure
arithmetic with no scene dependence, so it can be re-derived here exactly; a stored image
would pin this build's answer rather than Cycles'. The transcription is the reference and
the plugin is the thing under test -- never the other way round.

THREE CONTROLS, because a comparison against a term that is 1 everywhere passes for free:
  * VACUITY -- a quarter of the sweep must have a non-identity term, and at least one case
    must be a hard rejection, or the `keep` half is untested;
  * TOLERANCE -- the single-precision tolerance must still reject a 1% error in
    `bump_alpha2`, which is checked by measuring against a deliberately wrong reference;
  * WIRING -- `eval()` with the flag on and off must actually differ, or the property the
    exporter sets reaches nothing.

A NOTE FOR ANYONE EDITING THE PLUGIN: `import mitsuba` resolves to the BUILT package, not
to `src/python/python/...`. Editing the checkout and re-running pytest exercises the OLD
code and passes.
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


# --- intern/cycles/kernel/closure/bsdf.h, bump_shadowing_term() -------------------------
def cy_bump_shadowing_term(Ns, Nc, I, is_eval, is_diffuse, use_correction, alpha_scale=1.0):
    if np.allclose(Nc, Ns):
        return 1.0
    cosNsI, cosNsN, cosNI = np.dot(Ns, I), np.dot(Ns, Nc), np.dot(Nc, I)
    if cosNsI * cosNsN * cosNI < 0.0 and (is_eval or is_diffuse):
        return 0.0
    if not is_diffuse:
        return 1.0
    if not use_correction:
        return 1.0
    cos_i, cos_d = abs(cosNsI), abs(cosNsN)
    if cos_d >= 1.0 or cos_i >= 1.0:
        return 1.0
    if cos_i < 1e-6:
        return 0.0
    tan2_d = 1.0 / cos_d ** 2 - 1.0
    bump_alpha2 = min(max(0.125 * tan2_d, 0.0), 1.0) * alpha_scale
    # bsdf_G<GGX>(alpha2, cos_N) = 1 / (1 + lambda); Heitz eq. 72.
    lam = 0.5 * (np.sqrt(1.0 + bump_alpha2 * max(1.0 / cos_i ** 2 - 1.0, 0.0)) - 1.0)
    return 1.0 / (1.0 + lam)


def _sph(theta, phi=0.0):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], dtype=float)


def _v3(a):
    # mi.Vector3f rejects np.float64 elements.
    return mi.Vector3f(*[float(x) for x in a])


# Normal-map deviations x incident angles, the latter reaching past the horizon so the
# hard-rejection branch is exercised rather than assumed.
CASES = [(tilt, ang, phi)
         for tilt in (0.0, 0.05, 0.2, 0.5, 0.9, 1.3)
         for ang in (0.05, 0.6, 1.2, 1.4, 1.5, 1.56, 1.9, 2.6)
         for phi in (0.0, 2.0)]

NS = np.array([0.0, 0.0, 1.0])


def _tol():
    # `cos_i` enters as `1/cos_i^2`, which is ~8.6e3 at the grazing end of the sweep, so a
    # single-precision variant differs from the float64 reference in the 6th digit for
    # reasons that are not algebra. test03 proves this is still tight enough to bite.
    return 1e-9 if dr.type_v(mi.Float) == dr.VarType.Float64 else 3e-5


def _sweep(alpha_scale=1.0, use_corr=True):
    from mitsuba.python.ad.bsdfs.blender_principled import BlenderPrincipledBSDF
    bare = BlenderPrincipledBSDF.__new__(BlenderPrincipledBSDF)
    bare.bump_map_correction = use_corr
    worst, nonidentity, rejected = 0.0, 0, 0
    for tilt, ang, phi in CASES:
        Nc, I = _sph(tilt, 0.3), _sph(ang, phi)
        si = dr.zeros(mi.SurfaceInteraction3f, 1)
        keep, soft = bare._bump_shadowing(si, mi.Frame3f(_v3(Nc)), _v3(I), True)
        got = float(soft[0]) * (1.0 if bool(keep[0]) else 0.0)
        exp = cy_bump_shadowing_term(NS, Nc, I, True, True, use_corr,
                                     alpha_scale=alpha_scale)
        worst = max(worst, abs(got - exp))
        nonidentity += exp < 0.999
        rejected += exp == 0.0
    return worst, nonidentity, rejected


def test01_shadowing_term_matches_the_kernel():
    mi.set_variant(_variant())
    worst_on, _, _ = _sweep(use_corr=True)
    worst_off, _, _ = _sweep(use_corr=False)
    tol = _tol()
    assert worst_on < tol, (
        "correction ON: worst |mitsuba - cycles| over %d cases is %.3e, tolerance %.1e"
        % (len(CASES), worst_on, tol))
    assert worst_off < tol, (
        "correction OFF: worst |mitsuba - cycles| over %d cases is %.3e, tolerance %.1e. "
        "The HARD REJECTION is not gated by the material flag, so OFF is not the identity."
        % (len(CASES), worst_off, tol))


def test02_the_sweep_is_not_vacuous():
    """A term that were 1 everywhere would agree with any implementation."""
    mi.set_variant(_variant())
    _, nonidentity, rejected = _sweep(use_corr=True)
    assert nonidentity > len(CASES) // 4, (
        "only %d of %d cases have a non-identity term -- test01 would pass for a stub that "
        "returned 1" % (nonidentity, len(CASES)))
    assert rejected > 0, (
        "no case in the sweep is hard-rejected, so the `keep` half of the term is untested")


def test03_the_tolerance_still_discriminates():
    """Measure against a reference whose `bump_alpha2` is 1% wrong; it must NOT pass."""
    mi.set_variant(_variant())
    worst, _, _ = _sweep(alpha_scale=1.01)
    tol = _tol()
    assert worst > tol, (
        "a 1%% error in bump_alpha2 moves the worst case only %.3e, which is inside the "
        "%.1e tolerance -- test01 cannot detect a real error" % (worst, tol))


def _bsdf(correction):
    return mi.load_dict({
        'type': 'blender_principled',
        'two_sided': True, 'multiscatter': False,
        'base_color': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]},
        'roughness': 1.0, 'metallic': 0.0, 'eta': 1.45, 'alpha': 1.0,
        'diffuse_roughness': 0.0,
        'spec_ior_level': 0.0, 'spec_tint': {'type': 'rgb', 'value': [1, 1, 1]},
        'anisotropic': 0.0, 'anisotropic_rot': 0.0, 'transmission': 0.0,
        'clearcoat': 0.0, 'clearcoat_roughness': 0.03, 'clearcoat_ior': 1.5,
        'clearcoat_tint': {'type': 'rgb', 'value': [1, 1, 1]},
        'sheen': 0.0, 'sheen_roughness': 0.5,
        'sheen_tint': {'type': 'rgb', 'value': [1, 1, 1]},
        # A CONSTANT tangent-space normal, encoded 0..1 the way Blender does: no height
        # field, no differencing, no footprint -- the bump machinery is entirely out, and
        # `sc->N != sd->N` is the term's only trigger.
        'normalmap': {'type': 'rgb', 'value': [0.5 + 0.5 * 0.45, 0.5, 0.5 + 0.5 * 0.893]},
        'bump_map_correction': correction,
    })


def test04_the_material_flag_reaches_eval():
    """WIRING: the property the exporter sets must change the rendered value."""
    mi.set_variant(_variant())
    on, off = _bsdf(True), _bsdf(False)

    si = dr.zeros(mi.SurfaceInteraction3f, 1)
    si.p = mi.Point3f(0, 0, 0)
    si.n = mi.Normal3f(0, 0, 1)
    # The FULL frame, not just its normal: `dr.zeros` leaves `sh_frame.s`/`.t` at zero and
    # assigning `.n` does not fill them. `eval` survives a degenerate frame and `sample`
    # does not, so this line is load-bearing the moment anyone adds a sampling assertion.
    si.sh_frame = mi.Frame3f(mi.Normal3f(0, 0, 1))
    si.dp_du, si.dp_dv = mi.Vector3f(1, 0, 0), mi.Vector3f(0, 1, 0)
    si.wi = _v3(_sph(0.5))
    ctx = mi.BSDFContext()

    Nc = _sph(np.arcsin(0.45), 0.0)
    moved, checked = 0, 0
    for ang in (0.2, 0.9, 1.3, 1.45, 1.52, 1.55, 1.565):
        # phi = 0 is the same azimuth as the normal-map tilt, so the lobe survives the
        # shading horizon and it is the SOFTENING being read, not the hard rejection.
        wo_np = _sph(ang, 0.0)
        wo = _v3(wo_np)
        v_on = float(mi.luminance(on.eval(ctx, si, wo))[0])
        v_off = float(mi.luminance(off.eval(ctx, si, wo))[0])
        if v_off <= 0.0:
            continue                      # hard rejection: zero in both arms, as it should be
        checked += 1
        exp = cy_bump_shadowing_term(NS, Nc, wo_np, True, True, True)
        ratio = v_on / v_off
        assert abs(ratio - exp) < 1e-4, (
            "at theta_o=%.3f the on/off ratio is %.6f but the kernel says %.6f"
            % (ang, ratio, exp))
        moved += abs(ratio - 1.0) > 1e-4

    assert checked >= 3, "too few surviving directions to test the wiring"
    assert moved >= 3, (
        "only %d direction(s) changed between correction on and off -- the "
        "`bump_map_correction` property is inert and the exporter flag reaches nothing"
        % moved)
