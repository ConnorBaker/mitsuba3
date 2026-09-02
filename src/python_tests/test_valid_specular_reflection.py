"""`blender_principled` against Cycles' `ensure_valid_specular_reflection` (piece 3 of
the bump-map correction, `kernel/closure/bsdf_util.h`).

WHAT THIS PINS. When shading normals (a normal map, or plain smooth shading) push the
specular reflection of the view ray below the geometric surface, Cycles rotates the
closure normal toward `Ng` -- just enough that the reflection clears the surface by a
threshold `min(0.9 * dot(Ng, I), 0.01)` -- and hands that normal to every GLOSSY lobe
(`maybe_ensure_valid_specular_reflection` in `svm/closure.h`: metallic, specular,
transmission; NOT diffuse/sheen; the coat has its own copy). The port must fire on the
same configurations, produce the same normal, and keep sample() and pdf() describing the
same distribution -- including on two-sided BACK faces, where the pre-mirror sign of
`si.wi.z` is the only way to reconstruct which side the geometric normal was on.

WHY A TRANSCRIPTION RATHER THAN A GOLDEN IMAGE. Same reasoning as
`test_bump_map_correction`: the kernel function is pure arithmetic with no scene
dependence, so the reference is re-derived here exactly and the plugin is the thing under
test.

CONTROLS, because each probe has a way of passing for free:
  * VACUITY -- a fraction of the sweep must actually trigger the correction;
  * REACH -- a diffuse-only material must be BIT-identical with the correction frame
    forced to identity (piece 3 must not leak into diffuse), while a glossy material at
    grazing incidence must differ (the wiring reaches eval);
  * MIS -- on a two-sided back face, `sample().pdf` must equal a fresh `pdf()` query, and
    a deliberately broken pre-mirror sign must make that check FAIL (the negative control
    that proves the check can see the bug it guards).
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


# --- intern/cycles/kernel/closure/bsdf_util.h, ensure_valid_specular_reflection() ------
def cy_ensure_valid_specular_reflection(Ng, I, N):
    R = 2.0 * np.dot(N, I) * N - I
    threshold = min(0.9 * np.dot(Ng, I), 0.01)
    if np.dot(Ng, R) >= threshold:
        return N
    NdotNg = np.dot(N, Ng)
    X = N - NdotNg * Ng
    lX = np.linalg.norm(X)
    X = X / lX if lX > 0.0 else N.copy()
    Ix, Iz = np.dot(I, X), np.dot(I, Ng)
    a = Ix * Ix + Iz * Iz
    b = 2.0 * (a + Iz * threshold)
    c = (threshold + Iz) ** 2
    disc = np.sqrt(max(b * b - 4.0 * a * c, 0.0))
    Nz2 = 0.25 * (b + disc) / a if Ix < 0.0 else 0.25 * (b - disc) / a
    Nz2 = min(max(Nz2, 0.0), 1.0)
    Nx = np.sqrt(1.0 - Nz2)
    Nz = np.sqrt(Nz2)
    return Nx * X + Nz * Ng


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


def _bsdf(nmap_color=None, **kw):
    d = {"type": "blender_principled",
         "base_color": {"type": "rgb", "value": [0.9, 0.9, 0.9]},
         "roughness": 0.2, "metallic": 1.0}
    if nmap_color is not None:
        d["normalmap"] = {"type": "rgb", "value": list(nmap_color)}
    d.update(kw)
    return mi.load_dict(d)


def test01_corrected_normal_matches_the_kernel():
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs.common import compute_normalmap_frame

    rng = np.random.default_rng(7)
    fired = 0
    for _ in range(48):
        # tangent-space closure normal (upper hemisphere) and a view direction
        n = rng.normal(size=3)
        n[2] = abs(n[2]) + 0.05
        n /= np.linalg.norm(n)
        wi = rng.normal(size=3)
        wi[2] = abs(wi[2]) * rng.uniform(0.02, 1.0)  # bias toward grazing
        wi /= np.linalg.norm(wi)

        bsdf = _bsdf(nmap_color=(0.5 * (n + 1.0)).tolist())
        si = _make_si(tuple(wi))
        attr = bsdf.fetch_attributes(si, mi.Bool(True))
        frame = compute_normalmap_frame(si, attr.normal, rot=attr.anisotropic_rot)
        g = bsdf._valid_reflection_frame(si, frame, mi.Float(si.wi.z), attr)

        n_plugin = np.array([g.n.x[0], g.n.y[0], g.n.z[0]])
        n_frame = np.array([frame.n.x[0], frame.n.y[0], frame.n.z[0]])
        # sh_frame == geometry here, so local Ng is exactly +z
        n_ref = cy_ensure_valid_specular_reflection(
            np.array([0.0, 0.0, 1.0]), wi, n_frame)
        n_ref /= np.linalg.norm(n_ref)
        if not np.allclose(n_ref, n_frame):
            fired += 1
        assert np.allclose(n_plugin, n_ref, atol=2e-3), (
            f"n={n} wi={wi}: plugin {n_plugin} vs kernel {n_ref}")

    # VACUITY: the sweep must exercise the quartic, not just the early-out.
    assert fired >= 8, f"only {fired}/48 configs triggered the correction"


def test02_the_threshold_property_holds_where_it_fires():
    # The quartic's defining property: where the correction fires, the corrected
    # normal places the reflected view ray EXACTLY at the clearance threshold.
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs.common import compute_normalmap_frame

    bsdf = _bsdf(nmap_color=[0.9, 0.5, 0.8])            # N tilted to (0.8, 0, 0.6)
    si = _make_si((-0.995, 0.0, 0.0999))                # grazing against the tilt
    attr = bsdf.fetch_attributes(si, mi.Bool(True))
    frame = compute_normalmap_frame(si, attr.normal, rot=attr.anisotropic_rot)
    g = bsdf._valid_reflection_frame(si, frame, mi.Float(si.wi.z), attr)

    wi = si.wi
    r0 = 2.0 * dr.dot(frame.n, wi) * frame.n - wi
    thresh = dr.minimum(0.9 * wi.z, 0.01)
    assert bool((dr.dot(mi.Vector3f(0, 0, 1), r0) < thresh)[0]), "probe does not fire"
    r1 = 2.0 * dr.dot(g.n, wi) * g.n - wi
    assert abs(r1.z[0] - thresh[0]) < 1e-3
    assert abs(g.n.x[0] - frame.n.x[0]) > 1e-3   # and the frame genuinely moved


def test03_piece3_reaches_glossy_and_only_glossy():
    mi.set_variant(_variant())

    si = _make_si((-0.995, 0.0, 0.0999))
    wo = dr.normalize(mi.Vector3f(0.4, 0.2, 0.9))
    ctx = mi.BSDFContext()
    tilt = [0.9, 0.5, 0.8]

    ident = lambda self, si, frame, wiz0, attr, use_rot=True: frame

    # diffuse-only: identity-forced and real must agree exactly.
    # Patch `type(<instance>)`, not a module-path class -- the bsdfs package
    # double-imports under two module paths (see test04's control for the story).
    diff = _bsdf(nmap_color=tilt, metallic=0.0, spec_ior_level=0.0, roughness=0.9)
    cls = type(diff)
    orig = cls._valid_reflection_frame
    v1, p1 = diff.eval_pdf(ctx, si, wo)
    cls._valid_reflection_frame = ident
    v0, p0 = diff.eval_pdf(ctx, si, wo)
    cls._valid_reflection_frame = orig
    assert dr.max(dr.abs(v1 - v0))[0] == 0.0 and dr.abs(p1 - p0)[0] == 0.0

    # metallic at grazing: they must DIFFER (Cycles recovers a reflection the
    # uncorrected frame zeroes)
    met = _bsdf(nmap_color=tilt)
    assert type(met) is cls
    v1, _ = met.eval_pdf(ctx, si, wo)
    cls._valid_reflection_frame = ident
    v0, _ = met.eval_pdf(ctx, si, wo)
    cls._valid_reflection_frame = orig
    assert dr.max(dr.abs(v1 - v0))[0] > 1e-4


def test04_backface_sample_pdf_agrees_and_the_control_fails():
    mi.set_variant(_variant())

    # Smooth shading alone triggers the correction: sh_frame tilted off si.n, no
    # normal map, so wi stays above the closure normal and samples survive.
    ts = _bsdf(two_sided=True)
    si = _make_si((-0.995, 0.0, -0.0999), sh_n=(0.35, 0.1, 0.93))
    ctx = mi.BSDFContext()
    N = 64
    rng = mi.PCG32(size=N)
    s1 = rng.next_float32()
    s2 = mi.Point2f(rng.next_float32(), rng.next_float32())

    bs, _ = ts.sample(ctx, si, s1, s2, mi.Bool(True))
    pq = ts.pdf(ctx, si, bs.wo)
    act = bs.pdf > 0.0
    assert int(dr.count(act)[0]) >= 8, "probe vacuous: samples all rejected"
    rel = dr.select(act, dr.abs(bs.pdf - pq) / dr.maximum(bs.pdf, 1e-9), 0.0)
    assert dr.max(rel)[0] < 1e-4, "back-face MIS: sample and query pdf disagree"

    # NEGATIVE CONTROL: drop the pre-mirror sign and the same check must fail.
    # Patch `type(ts)`, not a module-path class: the bsdfs package double-imports as
    # `mitsuba.python.ad.bsdfs` AND `mitsuba.ad.bsdfs` (two distinct classes), and
    # which one `load_dict` instantiates depends on import order across the suite --
    # a module-path patch can silently miss the instance (caught in
    # `test_interior_closure_policy`, where the ggx-table tests flipped the binding).
    cls = type(ts)
    real = cls._eval_pdf_impl

    def broken(self, attr, ctx, si, wo_, active, is_eval=True, wiz0=None):
        return real(self, attr, ctx, si, wo_, active, is_eval=is_eval, wiz0=None)

    cls._eval_pdf_impl = broken
    try:
        bs2, _ = ts.sample(ctx, si, s1, s2, mi.Bool(True))
    finally:
        cls._eval_pdf_impl = real
    pq2 = ts.pdf(ctx, si, bs2.wo)
    rel2 = dr.select(bs2.pdf > 0.0,
                     dr.abs(bs2.pdf - pq2) / dr.maximum(bs2.pdf, 1e-9), 0.0)
    assert dr.max(rel2)[0] > 1e-3, (
        "negative control failed -- the MIS check cannot see the wiz0 bug it guards")


def test05_piece3_reaches_the_coat():
    """Cycles corrects the COAT with its own copy of the fix (`svm/closure.h`:
    `valid_coat_normal = maybe_ensure_valid_specular_reflection(sd, coat_normal)`), used
    for the coat closure AND the tint's optical depth. The port did not, for a while --
    the coat frame went uncorrected while the base glossy frame was fixed -- and test03
    could not see that, because its probes carry no coat. Same design as test03 on a
    coat-only configuration: the base is pure diffuse (test03 proves piece 3 is a no-op
    there), so any identity-forced difference is the coat's.
    """
    mi.set_variant(_variant())
    from mitsuba.python.ad.bsdfs.common import compute_normalmap_frame

    tilt = [0.9, 0.5, 0.8]                              # coat normal (0.8, 0, 0.6)
    si = _make_si((-0.995, 0.0, 0.0999))                # grazing against the tilt
    wo = dr.normalize(mi.Vector3f(0.4, 0.2, 0.9))
    ctx = mi.BSDFContext()

    coat = _bsdf(metallic=0.0, spec_ior_level=0.0, roughness=0.9,
                 clearcoat=1.0, clearcoat_roughness=0.3,
                 clearcoat_normalmap={"type": "rgb", "value": tilt})

    # VACUITY: the correction must actually fire on the COAT frame here.
    attr = coat.fetch_attributes(si, mi.Bool(True))
    cc_frame = compute_normalmap_frame(si, normal=attr.clearcoat_normal)
    g = coat._valid_reflection_frame(si, cc_frame, mi.Float(si.wi.z), attr,
                                     use_rot=False)
    assert abs(g.n.x[0] - cc_frame.n.x[0]) > 1e-3, "probe does not fire on the coat"

    # REACH: identity-forced vs real must differ -- the corrected coat frame is wired
    # into eval. Patch type(<instance>), per test03/test04's double-import story.
    ident = lambda self, si, frame, wiz0, attr, use_rot=True: frame
    cls = type(coat)
    orig = cls._valid_reflection_frame
    v1, p1 = coat.eval_pdf(ctx, si, wo)
    cls._valid_reflection_frame = ident
    v0, p0 = coat.eval_pdf(ctx, si, wo)
    cls._valid_reflection_frame = orig
    assert dr.max(dr.abs(v1 - v0))[0] > 1e-4, "coat never sees the corrected frame"
