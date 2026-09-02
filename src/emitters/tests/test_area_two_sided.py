"""A two-sided area light must be indistinguishable from a one-sided one whose normals
point the other way. That equivalence is the whole specification, and it is what these
tests check -- not the individual masks, which can each look right while the estimator
is still inconsistent.

The trap this is written against: `eval`, `sample_direction` and `pdf_direction` each
carry their own side test, and MIS combines all three. Relaxing only `eval` produces an
image that is not black -- so a naive "is it visible now?" test passes -- but whose NEE
and BSDF strategies disagree, which shows up as a wrong mean rather than as an error.
Comparing against the flipped-normal reference exercises all three at once.

Why this exists at all: Cycles' emission is two-sided. `intern/cycles/kernel/closure/emissive.h`
computes `cosNI = fabsf(dot(Ng, wi))` and emits wherever `cosNI > 0`, so a mesh lamp emits
from whichever face a ray arrives on. Mitsuba's area light masks on `cos_theta(si.wi) > 0`.
A Blender scene whose lamp normals point away therefore lights the room in one renderer and
not the other, with nothing logged in either.
"""

import pytest
import drjit as dr
import mitsuba as mi


def _scene(flip, two_sided):
    """A floor lit by a lamp ABOVE and OUTSIDE the frame.

    The lamp must not be visible to the camera, and getting that wrong is not a detail:
    a first version put it between the floor and the camera, where a one-sided lamp
    turned away from the floor still shines straight AT the camera. The "is it dark?"
    control then read 0.0206 instead of ~0, and the equality test was off by the lamp's
    own image rather than by anything about the emitter. With the lamp out of frame the
    only path to the sensor is floor illumination, which is the quantity under test.

    `flip=True` turns the lamp's normal AWAY from the floor -- the case a one-sided light
    cannot render and a two-sided one must. Position, area and solid angle are identical
    in both arms; only the orientation differs.
    """
    return mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 3},
        'sensor': {
            'type': 'perspective',
            'fov': 30,
            # Below the lamp and looking down at the floor, so the lamp sits ~34 degrees
            # off the view axis against a 15 degree half-angle -- comfortably out of frame.
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, -0.5, 4], target=[0, -1, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 32, 'height': 32,
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': 512, 'seed': 7},
        },
        'floor': {
            'type': 'rectangle',
            # Horizontal, normal +y (up).
            'to_world': mi.ScalarTransform4f().translate([0, -1, 0])
                                              .rotate([1, 0, 0], -90).scale(4),
            'bsdf': {'type': 'diffuse', 'reflectance': 0.8},
        },
        'lamp': {
            'type': 'rectangle',
            # Horizontal at y = 1.5; unflipped normal is -y, i.e. pointing DOWN at the floor.
            'to_world': mi.ScalarTransform4f().translate([0, 1.5, 0])
                                              .rotate([1, 0, 0], 90).scale(0.5),
            'flip_normals': flip,
            'emitter': {'type': 'area', 'radiance': 20.0, 'two_sided': two_sided},
        },
    })


def _mean(scene):
    import numpy as np
    return float(np.asarray(mi.render(scene)).mean())


def test01_two_sided_from_the_back_equals_one_sided_from_the_front(variants_all_rgb):
    """The specification, stated as an equality."""
    back_two_sided = _mean(_scene(flip=True,  two_sided=True))
    front_one_sided = _mean(_scene(flip=False, two_sided=False))
    # Independent sample streams, so this is a Monte Carlo comparison, not a bitwise one.
    assert back_two_sided == pytest.approx(front_one_sided, rel=2e-2), \
        (back_two_sided, front_one_sided)


def test02_one_sided_from_the_back_is_dark(variants_all_rgb):
    """The control. Without it, test01 could pass on a scene that is bright for some
    reason having nothing to do with the emitter's orientation."""
    dark = _mean(_scene(flip=True, two_sided=False))
    lit = _mean(_scene(flip=False, two_sided=False))
    assert lit > 0.05, lit
    assert dark < lit * 1e-3, (dark, lit)


def test03_two_sided_is_symmetric_under_flipping(variants_all_rgb):
    """A two-sided light cannot care which way its normals point."""
    a = _mean(_scene(flip=False, two_sided=True))
    b = _mean(_scene(flip=True,  two_sided=True))
    assert a == pytest.approx(b, rel=2e-2), (a, b)


def test04_default_is_one_sided(variants_all_rgb):
    """Off by default: a scene written for Mitsuba keeps Mitsuba's semantics."""
    e = mi.load_dict({'type': 'area', 'radiance': 1.0})
    assert 'two_sided = 0' in str(e) or 'two_sided = false' in str(e), str(e)


def test05_eval_is_relaxed_on_the_back_face(variants_all_rgb):
    """The narrow unit check behind test01: `eval` alone, on both faces."""
    for two_sided, expect_back in ((False, 0.0), (True, 1.0)):
        scene = mi.load_dict({
            'type': 'scene',
            'lamp': {'type': 'rectangle',
                     'emitter': {'type': 'area', 'radiance': 1.0,
                                 'two_sided': two_sided}},
        })
        emitter = scene.shapes()[0].emitter()
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = mi.Vector3f(0, 0, 1)      # front
        assert dr.allclose(mi.luminance(emitter.eval(si)), 1.0)
        si.wi = mi.Vector3f(0, 0, -1)     # back
        assert dr.allclose(mi.luminance(emitter.eval(si)), expect_back), two_sided


def test06_sampled_power_doubles(variants_all_rgb):
    """`sample_ray` must account for choosing a face: half the directional density, so
    twice the weight. If the side choice were added without the factor of two, a
    two-sided light would emit the same total power as a one-sided one -- half of it
    aimed the wrong way -- and every test above would still pass."""
    import numpy as np

    def power(two_sided):
        scene = mi.load_dict({
            'type': 'scene',
            'lamp': {'type': 'rectangle',
                     'emitter': {'type': 'area', 'radiance': 1.0,
                                 'two_sided': two_sided}},
        })
        emitter = scene.shapes()[0].emitter()
        n = 4096
        rng = mi.PCG32(size=n)
        u = lambda: mi.Point2f(rng.next_float32(), rng.next_float32())
        _, weight = emitter.sample_ray(0.0, rng.next_float32(), u(), u(), True)
        return float(np.asarray(mi.luminance(weight)).mean())

    one, two = power(False), power(True)
    assert two == pytest.approx(2.0 * one, rel=5e-2), (one, two)
