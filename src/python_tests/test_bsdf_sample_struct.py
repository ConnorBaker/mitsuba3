"""Every traversed field of `BSDFSample3f` must be valid from every construction path.

WHY THIS EXISTS. `BSDFSample3` is a `DRJIT_STRUCT`, so `ad_call` traverses each listed
field when a BSDF is invoked through a vectorized callable -- which is how every BSDF in a
JIT variant is invoked. Dr.Jit rejects a struct with an empty member outright:

    RuntimeError: ad_call(): callable 28 returned an empty/uninitialized Dr.Jit array,
    which is not allowed

The default constructor initializes NOTHING, which is fine for a field that every plugin
assigns and fatal for one that predates none of them. Adding `sampled_roughness_squared` to
the struct without a default member initializer broke every Python BSDF that builds its
sample as `mi.BSDFSample3f()` -- `translucent` and `refraction` both do -- and the symptom
was a whole-scene render failure with no mention of roughness anywhere in it.

WHAT THE INVARIANT ACTUALLY IS -- measured, because the first version of this test asserted
something stronger and false. `mi.BSDFSample3f()` leaves ALL SIX traversed fields empty, not
just the new one; `wo`, `pdf`, `eta`, `sampled_type` and `sampled_component` always have.
That is harmless only because every existing plugin assigns those five before returning. So
the property is not "no field is ever empty" -- it is BACKWARD COMPATIBILITY: a plugin
written against the field set that existed BEFORE a new field was added must still return a
fully-valid struct. The test below encodes exactly that, by assigning the historical five
and requiring nothing to be left empty. A test that only checked
`sampled_roughness_squared` would pin the bug that was found and miss the next one of the
same shape.
"""

import pytest

import drjit as dr
import mitsuba as mi


def _variant():
    for v in ("llvm_ad_rgb", "cuda_ad_rgb"):
        if v in mi.variants():
            return v
    pytest.skip("no JIT variant; the struct-traversal path only exists in one")


def _fields():
    """The struct's traversed field names, from the binding rather than from a literal."""
    ds = getattr(mi.BSDFSample3f, "DRJIT_STRUCT", None)
    if not ds:
        pytest.skip("BSDFSample3f exposes no DRJIT_STRUCT to enumerate")
    return list(ds.keys())


#: The fields a BSDF plugin written before `sampled_roughness_squared` existed knows to
#: assign. This is a historical fact about the struct, not something derivable from it, so
#: it is written down. Anything OUTSIDE this set must be valid without being assigned.
LEGACY_ASSIGNED = ('wo', 'pdf', 'eta', 'sampled_type', 'sampled_component')


def test01_a_legacy_plugin_still_returns_a_valid_struct():
    mi.set_variant(_variant())
    names = _fields()
    assert len(names) >= len(LEGACY_ASSIGNED), "field list looks truncated: %r" % (names,)
    for n in LEGACY_ASSIGNED:
        assert n in names, (
            "%r is no longer a traversed field; LEGACY_ASSIGNED is stale and this test is "
            "no longer testing what it claims" % n)

    # Exactly what `translucent.py` / `refraction.py` do: default-construct, assign the
    # fields the plugin knows about, return.
    bs = mi.BSDFSample3f()
    bs.wo = mi.Vector3f(0, 0, 1)
    bs.pdf = mi.Float(1.0)
    bs.eta = mi.Float(1.0)
    bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DiffuseReflection)
    bs.sampled_component = mi.UInt32(0)

    empty = [n for n in names if dr.width(getattr(bs, n)) == 0]
    assert not empty, (
        "a plugin that assigns only the pre-existing fields %r still leaves %r EMPTY. "
        "`ad_call` rejects such a struct with 'returned an empty/uninitialized Dr.Jit "
        "array', taking down renders that have nothing to do with the new field. Give it a "
        "default member initializer in `include/mitsuba/render/bsdf.h`."
        % (list(LEGACY_ASSIGNED), empty))


def test02_zeros_and_default_ctor_agree():
    """The two zero-ish construction paths must produce the same sentinel values.

    `dr.zeros` fills every field; the default constructor relies on member initializers.
    If they disagree, a sentinel means different things depending on how a plugin happened
    to build its sample -- which is exactly what the sentinel design forbids.
    """
    mi.set_variant(_variant())
    a, b = mi.BSDFSample3f(), dr.zeros(mi.BSDFSample3f)
    disagree = []
    for n in _fields():
        va, vb = getattr(a, n), getattr(b, n)
        if dr.width(va) == 0:
            continue  # test01 owns that failure; do not report it twice
        if dr.any(dr.ravel(va) != dr.ravel(vb)):
            disagree.append((n, va, vb))
    assert not disagree, (
        "default-constructed and dr.zeros-constructed samples differ in %r" % (disagree,))


def test03_a_python_bsdf_survives_a_vectorized_call():
    """The end-to-end failure: a Python plugin that uses the bare default constructor.

    `translucent` builds its sample as `mi.BSDFSample3f()`, so it is the plugin the missing
    initializer actually broke. Calling it through `dr.dispatch`-backed BSDF machinery on a
    wide input is what exercises the struct traversal; a scalar call does not.
    """
    mi.set_variant(_variant())
    try:
        bsdf = mi.load_dict({'type': 'translucent'})
    except Exception as e:
        pytest.skip("translucent plugin unavailable: %s" % e)

    n = 64
    si = dr.zeros(mi.SurfaceInteraction3f, n)
    si.p = mi.Point3f(0, 0, 0)
    si.n = mi.Normal3f(0, 0, 1)
    si.sh_frame = mi.Frame3f(mi.Normal3f(0, 0, 1))
    si.wi = mi.Vector3f(0, 0, 1)
    si.wavelengths = mi.Color0f()

    bs, weight = bsdf.sample(mi.BSDFContext(), si, dr.full(mi.Float, 0.5, n),
                             mi.Point2f(0.3, 0.6))
    dr.eval(bs, weight)
    for fname in _fields():
        v = getattr(bs, fname)
        assert dr.width(v) > 0, (
            "translucent.sample() returned a BSDFSample whose %r is empty; `ad_call` "
            "refuses such a struct and the render dies far from this plugin" % fname)
