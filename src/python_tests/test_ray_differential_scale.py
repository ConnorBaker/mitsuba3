"""`RayDifferential3f.diff_scale` must be readable from Python AND survive struct traversal.

WHAT diff_scale IS FOR. The sensor emits a one-pixel differential and
`SamplingIntegrator::render` immediately shrinks it by `rsqrt(spp)`, so a stochastic average
over the pixel's samples reconstructs a pixel-wide filter. That is right for a LINEAR
consumer (a mip-mapped texture fetch) and wrong for a non-linear one -- the mean of N
perturbed shading normals is not the normal you get from perturbing once over the whole
pixel. `diff_scale` records the accumulated factor so such a consumer can divide it out and
recover the one-pixel footprint, which is what Cycles carries unconditionally.

TWO WAYS THAT BREAKS, and this file pins both.

1. The field was bound NOWHERE. `scale_differential` was callable from Python but the factor
   it accumulated was unreadable, so the footprint could not be checked from a harness at
   all.

2. Worse and quieter: the C++ `DRJIT_STRUCT` in `core/ray.h` traversed `diff_scale` while
   `MI_PY_DRJIT_STRUCT` in `core/python/ray_v.cpp` did not. The two must agree. While they
   disagreed, any Python-side traversal -- `dr.select`, a gather, or the loop state of a
   `dr.while_loop`, which is exactly how a Python integrator carries a ray -- dropped the
   field and reset it to its default with NO error. A consumer that divided by it would get
   the `rsqrt(spp)`-shrunk footprint back and no indication anything had happened.

MEASURED, not assumed. `has_differentials` is bound as a field but is NOT in the struct list
(it is a plain C++ bool, so it cannot be), which makes it the control: under `dr.select` it
goes True -> False silently, while `diff_scale` carries its value. That is the mechanism in
(2), demonstrated rather than argued.
"""

import pytest

import drjit as dr
import mitsuba as mi


def _variant():
    for v in ("llvm_ad_rgb", "cuda_ad_rgb"):
        if v in mi.variants():
            return v
    pytest.skip("no JIT variant; struct traversal only exists in one")


def test01_diff_scale_is_readable_and_accumulates():
    """`scale_differential` multiplies into it, so two calls compose."""
    mi.set_variant(_variant())
    r = mi.RayDifferential3f(mi.Point3f(0, 0, 0), mi.Vector3f(0, 0, 1))
    assert hasattr(r, "diff_scale"), (
        "RayDifferential3f exposes no `diff_scale`; the factor `render()` applies to the "
        "differentials is then unreadable from Python and no harness can check the footprint")
    assert dr.allclose(r.diff_scale, 1.0), "a fresh ray should start at diff_scale 1"
    r.scale_differential(0.5)
    r.scale_differential(0.25)
    assert dr.allclose(r.diff_scale, 0.125), (
        "diff_scale must ACCUMULATE (0.5 * 0.25 = 0.125), got %s" % (r.diff_scale,))


def test02_diff_scale_survives_struct_traversal():
    """The C++ and Python struct field lists must agree, or the field is silently reset."""
    mi.set_variant(_variant())
    fields = set(mi.RayDifferential3f.DRJIT_STRUCT.keys())
    assert "diff_scale" in fields, (
        "`diff_scale` is missing from the Python DRJIT_STRUCT list %r while `core/ray.h` "
        "traverses it. Any dr.select / gather / while_loop over a RayDifferential3f will "
        "drop it and silently reset it to 1." % sorted(fields))

    n = 8
    r = dr.zeros(mi.RayDifferential3f, n)
    r.o, r.d = mi.Point3f(0, 0, 0), mi.Vector3f(0, 0, 1)
    r.diff_scale = dr.full(mi.Float, 0.25, n)

    out = dr.select(dr.full(mi.Bool, True, n), r, dr.zeros(mi.RayDifferential3f, n))
    dr.eval(out)
    assert dr.all(dr.ravel(out.diff_scale) == 0.25), (
        "dr.select dropped diff_scale: expected 0.25 throughout, got %s" % (out.diff_scale,))


def test03_the_traversal_test_is_not_vacuous():
    """A field ABSENT from the struct list must actually be lost, or test02 proves nothing.

    If omission were harmless, test02 would pass whether or not `diff_scale` were listed and
    would be measuring nothing. `has_differentials` is the control -- bound, but not in the
    list -- and it must NOT survive.
    """
    mi.set_variant(_variant())
    assert "has_differentials" not in mi.RayDifferential3f.DRJIT_STRUCT, (
        "the control field is now IN the struct list; pick another unlisted field or this "
        "test no longer establishes that omission costs anything")

    n = 8
    r = dr.zeros(mi.RayDifferential3f, n)
    r.o, r.d = mi.Point3f(0, 0, 0), mi.Vector3f(0, 0, 1)
    r.has_differentials = True

    out = dr.select(dr.full(mi.Bool, True, n), r, dr.zeros(mi.RayDifferential3f, n))
    dr.eval(out)
    assert not out.has_differentials, (
        "an UNLISTED field survived dr.select, so omission from DRJIT_STRUCT costs nothing "
        "and test02 is vacuous -- re-derive what that test is actually pinning")
