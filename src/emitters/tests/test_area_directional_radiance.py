"""An `area` emitter must tell its radiance texture WHICH DIRECTION it is emitting in.

Every `SurfaceInteraction` the emitter handed to `m_radiance->eval()` outside of `eval()`
itself used to carry `wi = 0`: the `PositionSample` constructor initialises it to zero
explicitly, and `eval_parameterization` never sets it. A texture that reads `si.uv` -- which
is every ordinary one -- cannot tell. A texture whose value DEPENDS on the emitted direction
is destroyed by it, and destroyed asymmetrically: rays that happen to hit the emitter go
through `eval()` and see a real `wi`, while every next-event-estimation sample sees the zero
vector. NEE carries almost all of the light, so the emitter went black while still looking
alive in the one strategy that does not matter.

Found through Cycles' area-light SPREAD attenuation, which is exactly such a texture. A
45-degree spread lamp in a closed grey box rendered at 0.000069 against Cycles' 0.085087 --
a ratio of 0.0008, with Mitsuba's brightest pixel anywhere at 0.00149.

The texture below reports the cosine of the emitted direction, so the emitter's answer can be
checked against geometry the caller already holds: for a flat emitter the shading normal IS
`ds.n`, so it must equal `dot(-ds.d, ds.n)`. Both of the emitter's sampling strategies are
exercised, because `is_spatially_varying()` selects between two completely different
constructions of that interaction and only one of them was ever suspected.
"""

import pytest
import drjit as dr
import mitsuba as mi


#: variant name -> the texture class defined for it. Two constraints collide here and this
#: dict is what satisfies both. `mi.Texture` cannot be subclassed at import time ("Cannot
#: access 'Texture' before setting a variant"), so the class must be built lazily inside a
#: function; but building a FRESH class for every parametrized case and registering each one
#: under the same plugin name leaves the registry holding stale class objects, and the next
#: `load_dict` segfaults rather than complaining. So: exactly one class, and one
#: registration, per variant.
_PLUGIN = {}


def _ensure_registered():
    variant = mi.variant()
    if variant in _PLUGIN:
        return

    class WiCosine(mi.Texture):
        """Radiance equal to the cosine of the emitted direction, `si.wi.z`.

        `varying` selects which branch of `area::sample_direction` builds the interaction. It
        is a plugin PROPERTY rather than a closure variable so that one class can serve both.
        """

        def __init__(self, props):
            mi.Texture.__init__(self, props)
            self._varying = bool(props.get('varying', True))

        def eval(self, si, active=True):
            return mi.Color3f(si.wi.z)

        def eval_1(self, si, active=True):
            return mi.Float(si.wi.z)

        def eval_3(self, si, active=True):
            return mi.Color3f(si.wi.z)

        def mean(self):
            return 0.5

        def is_spatially_varying(self):
            return self._varying

        def to_string(self):
            return f'WiCosine[varying={self._varying}]'

    mi.register_texture('test_wi_cosine', lambda props: WiCosine(props))
    _PLUGIN[variant] = WiCosine


def _emitter(varying):
    """A unit rectangle in the z = 0 plane, emitting towards +z.

    Returns the SHAPE as well, and every caller must keep it alive. `Shape::emitter()` hands
    back a non-owning pointer and the emitter stores `m_shape` the same way, so letting the
    shape fall out of scope leaves `sample_direction` dereferencing a destroyed object -- a
    segfault, not an exception. An earlier version of this file returned the emitter alone
    and crashed here rather than reporting anything.
    """
    _ensure_registered()
    shape = mi.load_dict({
        'type': 'rectangle',
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'test_wi_cosine', 'varying': bool(varying)},
        },
    })
    return shape, shape.emitter()


def _receiver(p):
    it = dr.zeros(mi.Interaction3f)
    it.p = mi.Point3f(*p)
    it.t = 0.0
    it.time = 0.0
    return it


def _floats(x):
    """Every element of `x` as Python floats, scalar and vectorised variants alike.

    `dr.min(..., axis=None)` reduces a vectorised array to width one but still returns a
    Dr.Jit `Float`, not a Python number, so `float()` on it raises TypeError. Iterating works
    for both, and a scalar variant's channel is not iterable, hence the fallback.
    """
    try:
        return [float(v) for v in x]
    except TypeError:
        return [float(x)]


def _min_component(spec):
    """Smallest value across all three channels and all lanes."""
    return min(min(_floats(spec[i])) for i in range(3))


def _max_component(spec):
    """Largest value across all three channels and all lanes."""
    return max(max(_floats(spec[i])) for i in range(3))


@pytest.mark.parametrize('varying', [False, True])
@pytest.mark.parametrize('p', [(0.0, 0.0, 3.0), (1.5, -0.5, 2.0)])
def test01_sample_direction_supplies_the_emitted_direction(variants_vec_backends_once_rgb,
                                                           varying, p):
    """`sample_direction`'s radiance must be evaluated with the real emitted direction."""
    shape, emitter = _emitter(varying)   # `shape` is load-bearing -- see `_emitter`
    it = _receiver(p)
    ds, weight = emitter.sample_direction(it, mi.Point2f(0.3, 0.7), True)

    # `ds.d` points from the receiver TO the emitter, so the emitted direction is its
    # negation, and for a flat emitter the shading normal is `ds.n`.
    expected = dr.dot(-ds.d, ds.n)
    assert dr.all(expected > 0.0, axis=None), "receiver placed on the emitter's dark side"

    # `sample_direction` returns radiance / pdf.
    radiance = weight * ds.pdf
    assert dr.allclose(radiance, mi.Color3f(expected), atol=1e-5), \
        f"got {radiance}, expected {expected} (varying={varying})"

    # The defect this file exists for produced EXACTLY zero, so say so directly rather than
    # leaving it to the comparison above to notice.
    assert _min_component(radiance) > 1e-3, \
        "radiance evaluated to zero -- si.wi was not supplied"


@pytest.mark.parametrize('varying', [False, True])
def test02_eval_direction_agrees_with_sample_direction(variants_vec_backends_once_rgb,
                                                       varying):
    """The MIS half must see the same direction as the sampling half.

    These are two strategies for the same integral; if they disagree about `wi` the estimator
    is inconsistent rather than merely dark, which is the harder failure to spot downstream.
    """
    shape, emitter = _emitter(varying)   # `shape` is load-bearing -- see `_emitter`
    it = _receiver((0.4, 0.2, 2.5))
    ds, weight = emitter.sample_direction(it, mi.Point2f(0.25, 0.6), True)

    from_sample = weight * ds.pdf
    from_eval = emitter.eval_direction(it, ds, True)
    assert dr.allclose(from_sample, from_eval, atol=1e-5), \
        f"sample_direction {from_sample} vs eval_direction {from_eval} (varying={varying})"
    assert _min_component(from_eval) > 1e-3, \
        "eval_direction evaluated to zero -- si.wi was not supplied"


def test03_the_check_can_fail(variants_vec_backends_once_rgb):
    """VACUITY CONTROL: the assertions above must be capable of failing.

    A receiver behind a one-sided emitter is the one configuration where zero is the CORRECT
    answer, so this pins that the tests read a real quantity rather than a constant that
    happens to be positive. Without it, a texture returning a fixed 1.0 would pass everything.
    """
    shape, emitter = _emitter(True)   # `shape` is load-bearing -- see `_emitter`
    it = _receiver((0.0, 0.0, -3.0))     # behind the rectangle
    ds, weight = emitter.sample_direction(it, mi.Point2f(0.3, 0.7), True)
    assert _max_component(weight) == 0.0, \
        "a one-sided area emitter lit its own back face"
