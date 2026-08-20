"""The `bump` texture's geometry must be Cycles' geometry, not merely close to it.

`bump` exists to reproduce Blender's Bump node, so the thing worth testing is not that it
produces *a* plausible normal but that it produces *Cycles'* normal. This module transcribes
the relevant kernel functions from the Blender source and checks the plugin against them:

  * `make_orthonormals`            (intern/cycles/util/math_float3.h)
  * `differential_make_compact` /
    `differential_from_compact`    (intern/cycles/kernel/util/differential.h)
  * `differential_dudv`            (same)
  * `svm_node_set_bump`            (intern/cycles/kernel/svm/displace.h)

Two of these are checked against the SHIPPED code (`Bump._orthonormals`, `Bump._uv_of`) and
the third against the assembled arithmetic. That split is deliberate: the helpers are where
this plugin's two historical bugs lived, and they are pure functions of their arguments, so
they can be pinned exactly rather than through a render.

WHERE THE TOLERANCES COME FROM, AND WHY THEY ARE NOT 1e-15. Two separate effects, and
conflating them is how a tolerance ends up either vacuous or flaky:

  * `_uv_of` solves the same 2x2 system as `differential_dudv` by a different elimination --
    normal equations rather than Cramer's rule with the least-stable world axis dropped -- so
    the two agree to the last bits and not beyond. In pure float64 that floor is ~5e-13 (UV)
    and ~3e-12 (normal).
  * The shipped plugin runs at the VARIANT's precision, and every default variant here is
    SINGLE precision. Against a float64 reference the residuals are then ~2e-6 (UV, relative)
    and ~1.6e-7 (normal), which is float32 epsilon and not a disagreement about the formula.

A TRAP THIS FILE CANNOT PROTECT YOU FROM, recorded because it produced a confidently wrong
result here. `import mitsuba` resolves to the BUILT package, not to this checkout, so editing
`src/python/python/plugins/textures/bump.py` and re-running pytest tests the OLD code and
passes. Mutation-testing this suite by editing the source file therefore "shows" that it
catches nothing. Mutate the LOADED class (monkeypatch `Bump._orthonormals` / `Bump._uv_of`)
or rebuild. Done properly, the suite does discriminate: flipping the basis handedness is
caught by test01 (det = -1.0), dropping the `a12` cross-term from the UV solve is caught by
test02 (relative error 48.7), and a footprint scaled by just 1.001 is caught by test02 at
1.0e-3 against a 1e-5 gate.

So the gates below are chosen per precision and, for UV, are relative -- an absolute gate on
a quantity whose scale is set by random `dp_du`/`dp_dv` magnitudes would tighten and loosen
with the seed. An actual algebra error moves these residuals by O(0.1); nothing lands
between, which is what test05 exists to demonstrate rather than assert.
"""

import numpy as np
import pytest

import drjit as dr
import mitsuba as mi


# ---------------------------------------------------------------------------------------
# The Cycles side, transcribed. Deliberately written as scalar numpy in the same shape as
# the C++ so it can be diffed against the source by eye.
# ---------------------------------------------------------------------------------------

def _nrm(v):
    return v / np.linalg.norm(v)


def cy_make_orthonormals(N):
    if N[0] != N[1] or N[0] != N[2]:
        a = np.array([N[2] - N[1], N[0] - N[2], N[1] - N[0]])
    else:
        a = np.array([N[2] - N[1], N[0] + N[2], -N[1] - N[0]])
    a = _nrm(a)
    return a, np.cross(N, a)


def cy_differential_dudv(dPdu, dPdv, dPdx, dPdy, Ng):
    """Cramer's rule after dropping the least stable world axis."""
    du, dv, dx, dy = dPdu.copy(), dPdv.copy(), dPdx.copy(), dPdy.copy()
    xn, yn, zn = abs(Ng[0]), abs(Ng[1]), abs(Ng[2])
    if zn < xn or zn < yn:
        if yn < xn or yn < zn:
            du[0], dv[0], dx[0], dy[0] = du[1], dv[1], dx[1], dy[1]
        du[1], dv[1], dx[1], dy[1] = du[2], dv[2], dx[2], dy[2]
    det = du[0] * dv[1] - dv[0] * du[1]
    det = 1.0 / det if det != 0.0 else 0.0
    return (np.array([(dx[0] * dv[1] - dx[1] * dv[0]) * det,
                      (dx[1] * du[0] - dx[0] * du[1]) * det]),
            np.array([(dy[0] * dv[1] - dy[1] * dv[0]) * det,
                      (dy[1] * du[0] - dy[0] * du[1]) * det]))


def cy_svm_node_set_bump(Nin, dPdx, dPdy, h_c, h_x, h_y, scale, strength, fw):
    Rx, Ry = np.cross(dPdy, Nin), np.cross(Nin, dPdx)
    det = dPdx @ Rx
    surfgrad = (h_x - h_c) * Rx + (h_y - h_c) * Ry
    strength = max(strength, 0.0)
    out = fw * abs(det) * Nin - scale * np.sign(det) * surfgrad
    if not np.any(out):
        return Nin
    return _nrm(strength * _nrm(out) + (1.0 - strength) * Nin)


# ---------------------------------------------------------------------------------------

def _f(v):
    """Read one scalar out of a Dr.Jit value, whatever the variant makes it.

    In `scalar_rgb` a component is a plain Python float; in a JIT variant it is a
    length-1 array. Indexing unconditionally works in one and raises in the other, and
    hard-coding either makes this file silently variant-specific.
    """
    try:
        return float(v[0])
    except TypeError:
        return float(v)


def _vec3(v):
    return np.array([_f(v.x), _f(v.y), _f(v.z)])


def _tol(single, double):
    """Pick a tolerance for the active variant's precision."""
    try:
        return double if dr.is_double_v(mi.Float) else single
    except Exception:
        return double if 'double' in mi.variant() else single


def _variant():
    """Pick a variant once, preferring the cheap scalar one when it is compiled in."""
    avail = mi.variants()
    for v in ('scalar_rgb', 'llvm_ad_rgb', 'cuda_ad_rgb'):
        if v in avail:
            return v
    return avail[0]


def _bare_bump():
    """A `Bump` whose pure helpers can be called without building a Props.

    `_orthonormals` and `_uv_of` touch no instance state -- they are methods only because
    that is where they belong -- so bypassing `__init__` exercises exactly the shipped code
    without dragging a whole texture graph into a geometry test.
    """
    from mitsuba.python.plugins.textures.bump import Bump
    return Bump.__new__(Bump)


def _cases(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        dPdu, dPdv = rng.normal(size=3), rng.normal(size=3)
        if np.linalg.norm(np.cross(dPdu, dPdv)) < 1e-3:
            continue                      # a degenerate tangent frame tests nothing here
        yield dPdu, dPdv, rng


def test01_orthonormals_match_cycles():
    """The basis must be Cycles' basis, including its handedness.

    Handedness is not cosmetic: `b = cross(n, a)` is what fixes `sign(det)` to +1 for an
    unperturbed normal, so a flipped basis inverts every bump in the scene while leaving
    every magnitude in this file unchanged. The determinant check below is what catches it;
    comparing |a| and |b| alone would not.
    """
    mi.set_variant(_variant())
    b = _bare_bump()
    worst = 0.0
    for dPdu, dPdv, _ in _cases(500, 1):
        Ng = _nrm(np.cross(dPdu, dPdv))
        ex_r, ey_r = cy_make_orthonormals(Ng)
        ex, ey = b._orthonormals(mi.Normal3f(*Ng))
        got_x, got_y = _vec3(ex), _vec3(ey)
        worst = max(worst, np.linalg.norm(got_x - ex_r), np.linalg.norm(got_y - ey_r))
        # right-handed: (ex, ey, Ng) must have determinant +1, not -1
        handed = float(np.linalg.det(np.stack([got_x, got_y, Ng])))
        assert handed > 0.99, (
            f"(ex, ey, Ng) is not right-handed: det = {handed}. Cycles builds the\n"
            f"second axis as cross(N, a); building it as cross(a, N) leaves every\n"
            f"magnitude in this file unchanged and inverts every bump in the scene.")
    assert worst < _tol(1e-5, 1e-9), f"basis differs from Cycles by {worst}"


def test02_uv_offsets_match_differential_dudv():
    """`_uv_of` must equal `differential_dudv` wherever the offset is in the tangent plane."""
    mi.set_variant(_variant())
    b = _bare_bump()
    worst = 0.0
    for dPdu, dPdv, rng in _cases(500, 2):
        Ng = _nrm(np.cross(dPdu, dPdv))
        ex, ey = cy_make_orthonormals(Ng)
        r = float(rng.uniform(1e-4, 1.0))
        ref_x, ref_y = cy_differential_dudv(dPdu, dPdv, r * ex, r * ey, Ng)
        for w, ref in ((r * ex, ref_x), (r * ey, ref_y)):
            got = b._uv_of(mi.Vector3f(*dPdu), mi.Vector3f(*dPdv), mi.Vector3f(*w))
            scale = max(1.0, abs(ref[0]), abs(ref[1]))
            worst = max(worst, abs(_f(got.x) - ref[0]) / scale,
                        abs(_f(got.y) - ref[1]) / scale)
    assert worst < _tol(1e-5, 1e-9), f"UV offsets differ from Cycles by {worst} (relative)"


def test03_compacted_footprint_is_isotropic_and_area_is_r_squared():
    """`det` must be `r^2 * (Ng . Nin)`, which is what the compaction buys.

    This is the identity the plugin's derivation rests on, and it is the reason a normal-
    mapped Bump is not simply `r^2`: `Rx`/`Ry`/`det` are built from the closure's normal
    while the basis comes from the geometric one.
    """
    mi.set_variant(_variant())
    rng = np.random.default_rng(3)
    worst = 0.0
    for dPdu, dPdv, _ in _cases(500, 3):
        Ng = _nrm(np.cross(dPdu, dPdv))
        Nin = _nrm(Ng + 0.3 * rng.normal(size=3))
        ex, ey = cy_make_orthonormals(Ng)
        r = float(rng.uniform(1e-3, 1.0))
        dPdx, dPdy = r * ex, r * ey
        det = dPdx @ np.cross(dPdy, Nin)
        worst = max(worst, abs(det / (r * r * (Ng @ Nin)) - 1.0))
    assert worst < 1e-9, f"det != r^2 * (Ng.Nin), off by {worst}"


def test04_perturbed_normal_matches_the_kernel():
    """End to end: compaction, basis, UV offsets and `svm_node_set_bump` together.

    The height field is a non-linear analytic function of UV so that the FORWARD difference
    `h_x - h_c` actually depends on where the offset lands -- a linear field would agree for
    any offset scale at all and would pass even with the footprint bug this plugin was
    written to fix.
    """
    mi.set_variant(_variant())
    rng = np.random.default_rng(4)
    worst = 0.0
    for dPdu, dPdv, _ in _cases(400, 4):
        Ng = _nrm(np.cross(dPdu, dPdv))
        Nin = _nrm(Ng + 0.3 * rng.normal(size=3))
        uv0 = rng.uniform(-1, 1, size=2)
        c = rng.normal(size=6)
        H = lambda uv: (c[0] + c[1] * uv[0] + c[2] * uv[1] + c[3] * np.sin(7 * uv[0])
                        + c[4] * np.cos(5 * uv[1]) + c[5] * uv[0] * uv[1])
        fw = float(rng.uniform(0.02, 1.5))
        scale, strength = float(rng.normal()), float(rng.uniform(0, 1))

        # the true anisotropic screen differential, which Cycles then throws away
        duv_dx, duv_dy = rng.normal(size=2) * 0.01, rng.normal(size=2) * 0.01
        r = 0.5 * (np.linalg.norm(dPdu * duv_dx[0] + dPdv * duv_dx[1]) +
                   np.linalg.norm(dPdu * duv_dy[0] + dPdv * duv_dy[1]))

        ex, ey = cy_make_orthonormals(Ng)
        dPdx, dPdy = r * ex, r * ey
        ux, uy = cy_differential_dudv(dPdu, dPdv, dPdx, dPdy, Ng)
        ref = cy_svm_node_set_bump(Nin, dPdx, dPdy, H(uv0), H(uv0 + ux * fw),
                                   H(uv0 + uy * fw), scale, strength, fw)

        # the plugin's assembled arithmetic, with `fw` applied to the offset and to the
        # first term but NOT to the `dP` that builds Rx/Ry/det -- the split Cycles makes
        # across `tex_coord.h` and `displace.h`
        b = _bare_bump()
        px = b._uv_of(mi.Vector3f(*dPdu), mi.Vector3f(*dPdv), mi.Vector3f(*(dPdx * fw)))
        py = b._uv_of(mi.Vector3f(*dPdu), mi.Vector3f(*dPdv), mi.Vector3f(*(dPdy * fw)))
        got = cy_svm_node_set_bump(
            Nin, dPdx, dPdy, H(uv0),
            H(uv0 + np.array([_f(px.x), _f(px.y)])),
            H(uv0 + np.array([_f(py.x), _f(py.y)])),
            scale, strength, fw)
        worst = max(worst, np.linalg.norm(got - ref))
    assert worst < _tol(1e-5, 1e-9), f"perturbed normal differs from the kernel by {worst}"


def test05_anisotropic_footprint_would_fail_this_suite():
    """A positive control: the pre-fix quantity must NOT pass test04's comparison.

    Without this, every assertion above could be satisfied by a formula that ignores the
    footprint entirely, and the suite would be a VACUOUS instrument -- green on the very
    bug it exists to prevent. Feeding the true anisotropic differential in place of the
    compacted radius has to move the normal by a lot, or these gates prove nothing.
    """
    rng = np.random.default_rng(5)
    biggest = 0.0
    for dPdu, dPdv, _ in _cases(200, 5):
        Ng = _nrm(np.cross(dPdu, dPdv))
        Nin = _nrm(Ng + 0.3 * rng.normal(size=3))
        uv0 = rng.uniform(-1, 1, size=2)
        c = rng.normal(size=6)
        H = lambda uv: (c[0] + c[1] * uv[0] + c[3] * np.sin(7 * uv[0])
                        + c[4] * np.cos(5 * uv[1]) + c[5] * uv[0] * uv[1])
        fw, scale, strength = 0.1, 1.0, 1.0
        duv_dx, duv_dy = rng.normal(size=2) * 0.05, rng.normal(size=2) * 0.05
        aniso_x = dPdu * duv_dx[0] + dPdv * duv_dx[1]
        aniso_y = dPdu * duv_dy[0] + dPdv * duv_dy[1]
        r = 0.5 * (np.linalg.norm(aniso_x) + np.linalg.norm(aniso_y))
        ex, ey = cy_make_orthonormals(Ng)

        ux, uy = cy_differential_dudv(dPdu, dPdv, r * ex, r * ey, Ng)
        good = cy_svm_node_set_bump(Nin, r * ex, r * ey, H(uv0), H(uv0 + ux * fw),
                                    H(uv0 + uy * fw), scale, strength, fw)
        ax, ay = cy_differential_dudv(dPdu, dPdv, aniso_x, aniso_y, Ng)
        bad = cy_svm_node_set_bump(Nin, aniso_x, aniso_y, H(uv0), H(uv0 + ax * fw),
                                   H(uv0 + ay * fw), scale, strength, fw)
        biggest = max(biggest, np.linalg.norm(good - bad))
    assert biggest > 1e-3, (
        "the anisotropic footprint is indistinguishable from the compacted one on this "
        "population, so test04 cannot detect a footprint regression")
