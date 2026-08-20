"""An `area` emitter must tell its radiance texture WHICH DIRECTION it is emitting in.

Every `SurfaceInteraction` the emitter hands to `m_radiance->eval()` outside of `eval()`
itself used to carry `wi = 0`: the `PositionSample` constructor initialises it to zero
explicitly, and `eval_parameterization` never sets it. A texture that reads `si.uv` -- which
is every ordinary one -- cannot tell. A texture whose value DEPENDS on the emitted direction
is destroyed by it, and destroyed asymmetrically: rays that happen to hit the emitter go
through `eval()` and see a real `wi`, while every next-event-estimation sample sees the zero
vector. NEE carries almost all of the light, so the emitter goes black while still looking
alive in the one strategy that does not matter.

Found through Cycles' area-light SPREAD attenuation, which is exactly such a texture. A
45-degree spread lamp in a closed grey box rendered at 0.000069 against Cycles' 0.085087 --
a ratio of 0.0008, with Mitsuba's brightest pixel anywhere at 0.00149.

The texture below reports the cosine of the emitted direction, so the value the emitter
computes can be checked against geometry the caller already has: for a flat emitter the
shading normal IS `ds.n`, so the answer must be `dot(-ds.d, ds.n)`. Both of the emitter's
sampling strategies are exercised, because `is_spatially_varying()` selects between two
completely different constructions of that interaction and only one of them was ever
suspected.
"""

import pytest
import drjit as dr
import mitsuba as mi


def _register(varying):
    """Register a radiance texture reporting `si.wi.z`, once per variant and per branch."""
    name = 'test_wi_cosine_varying' if varying else 'test_wi_cosine_uniform'

    class WiCosine(mi.Texture):
        def __init__(self, props):
            mi.Texture.__init__(self, props)

        def eval(self, si, active=True):
            return mi.Color3f(si.wi.z)

        def eval_1(self, si, active=True):
            return mi.Float(si.wi.z)

        def eval_3(self, si, active=True):
            return mi.Color3f(si.wi.z)

        def mean(self):
            return 0.5

        def is_spatially_varying(self):
            # Selects which branch of `area::sample_direction` builds the interaction.
            return varying

        def to_string(self):
            return f'WiCosine[varying={varying}]'

    try:
        mi.register_texture(name, lambda props: WiCosine(props))
    except Exception:
        # Already registered in this variant by an earlier test in the same session.
        pass
    return name


def _emitter(varying):
    name = _register(varying)
    # A unit rectangle in the z = 0 plane, emitting towards +z.
    shape = mi.load_dict({
        'type': 'rectangle',
        'emitter': {'type': 'area', 'radiance': {'type': name}},
    })
    return shape.emitter()


def _receiver(p):
    it = dr.zeros(mi.Interaction3f)
    it.p = mi.Point3f(*p)
    it.t = 0.0
    it.time = 0.0
    return it


@pytest.mark.parametrize('varying', [False, True])
@pytest.mark.parametrize('p', [(0.0, 0.0, 3.0), (1.5, -0.5, 2.0)])
def test01_sample_direction_supplies_the_emitted_direction(variants_vec_backends_once_rgb,
                                                           varying, p):
    """`sample_direction`'s radiance must be evaluated with the real emitted direction."""
    emitter = _emitter(varying)
    it = _receiver(p)
    ds, weight = emitter.sample_direction(it, mi.Point2f(0.3, 0.7), True)

    # `ds.d` points from the receiver TO the emitter, so the emitted direction is its
    # negation, and for a flat emitter the shading normal is `ds.n`.
    expected = dr.dot(-ds.d, ds.n)
    assert dr.all(expected > 0.0), "receiver placed on the emitter's dark side"

    # `sample_direction` returns radiance / pdf.
    radiance = weight * ds.pdf
    assert dr.allclose(radiance, mi.Color3f(expected), atol=1e-5), \
        f"got {radiance}, expected {expected} (varying={varying})"

    # The defect this file exists for produced EXACTLY zero, so say so directly rather than
    # relying on the comparison above to notice.
    assert dr.all(dr.mean(radiance, axis=None) > 1e-3), \
        "radiance evaluated to zero -- si.wi was not supplied"


@pytest.mark.parametrize('varying', [False, True])
def test02_eval_direction_agrees_with_sample_direction(variants_vec_backends_once_rgb,
                                                       varying):
    """The MIS half must see the same direction as the sampling half.

    These are two strategies for the same integral; if they disagree about `wi` the estimator
    is inconsistent rather than merely dark, which is the harder failure to spot downstream.
    """
    emitter = _emitter(varying)
    it = _receiver((0.4, 0.2, 2.5))
    ds, weight = emitter.sample_direction(it, mi.Point2f(0.25, 0.6), True)

    from_sample = weight * ds.pdf
    from_eval = emitter.eval_direction(it, ds, True)
    assert dr.allclose(from_sample, from_eval, atol=1e-5), \
        f"sample_direction {from_sample} vs eval_direction {from_eval} (varying={varying})"
    assert dr.all(dr.mean(from_eval, axis=None) > 1e-3), \
        "eval_direction evaluated to zero -- si.wi was not supplied"


def test03_the_check_can_fail(variants_vec_backends_once_rgb):
    """VACUITY CONTROL: the assertions above must be capable of failing.

    A receiver behind the emitter is the one configuration where zero is the CORRECT answer,
    so it pins that the tests are reading a real quantity rather than a constant that happens
    to be positive. Without this, a texture returning a fixed 1.0 would pass everything.
    """
    emitter = _emitter(True)
    it = _receiver((0.0, 0.0, -3.0))     # behind the rectangle
    ds, weight = emitter.sample_direction(it, mi.Point2f(0.3, 0.7), True)
    assert dr.all(dr.mean(weight, axis=None) == 0.0), \
        "a one-sided area emitter lit its own back face"
