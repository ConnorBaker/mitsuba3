"""Blender-style per-object ray visibility (\\ref RayMask).

Ported onto upstream's RayMask substrate: the ``visible_*`` shape properties clear bits of
``Shape.visibility_mask()``, integrators trace every ray with the bit(s) of the event that
spawned it, and the acceleration structures do the filtering (there is no scene-level walk
or gate any more).

Every behavioural assertion here is a RENDERED comparison against the same scene with the
switch left alone, because that is the only thing that establishes the feature does what its
name says: a property that parses, is stored, and is never tested by any ray would sail
through a construction-time check and change no pixel.

The scenes are built so that each switch is isolated BY GEOMETRY rather than by luck -- the
occluder in the shadow scene stands between the light and the floor but not between the floor
and the camera, and the one in the camera scene stands between the camera and the floor but
above the light. So a single flag is the only thing that can move the image.
"""

import pytest
import drjit as dr
import mitsuba as mi


def _base(res=32, spp=128):
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 4},
        'sensor': {
            'type': 'perspective',
            # Looking DOWN at the floor from one side, with a narrow field of view so every
            # camera ray descends: no ray can wander up into the occluder plane by accident.
            'fov': 20,
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, -6, 2],
                                                       target=[0, 0, 0],
                                                       up=[0, 0, 1]),
            'film': {'type': 'hdrfilm', 'width': res, 'height': res,
                     'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
            'sampler': {'type': 'independent', 'sample_count': spp},
        },
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale(4.0),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [1, 1, 1]}},
        },
    }


def _shadow_scene(occluder_props):
    """Point light overhead, opaque slab between it and the floor.

    A `point` emitter is deliberate: it is a delta light, so it contributes through next-event
    estimation ONLY and can never be seen by a camera ray. Whatever the image shows came
    through a shadow ray.
    """
    d = _base()
    d['light'] = {'type': 'point', 'position': [0, 0, 6],
                  'intensity': {'type': 'rgb', 'value': [200, 200, 200]}}
    occluder = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().translate([0, 0, 3]).scale(4.0),
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0, 0, 0]}},
    }
    occluder.update(occluder_props)
    d['occluder'] = occluder
    return mi.load_dict(d)


def _camera_scene(occluder_props):
    """Slab between the camera and the floor, with the light UNDERNEATH the slab.

    The light sits below the occluder plane, so no shadow ray ever reaches the occluder and
    the floor is lit in both arms; the only thing the flag can change is what primary rays
    see.

    The slab is deliberately WIDER than the floor. It has to be: the camera looks down at an
    angle, so the steepest rays in the frame cross the occluder plane further out than the
    shallowest ones, and a slab merely as wide as the floor lets the bottom of the frame see
    past its near edge -- which is 12% of the pixels, lit, in a frame the test wants black.
    """
    d = _base()
    d['light'] = {'type': 'point', 'position': [0, 0, 0.5],
                  'intensity': {'type': 'rgb', 'value': [50, 50, 50]}}
    occluder = {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().translate([0, 0, 1]).scale(12.0),
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0, 0, 0]}},
    }
    occluder.update(occluder_props)
    d['occluder'] = occluder
    return mi.load_dict(d)


def _camera_scene_without_occluder():
    """The same scene with the slab simply absent -- the value `visible_camera=False` must hit.

    Asserting only 'the image got brighter' would pass for a flag that made the slab merely
    grey. This pins the target.
    """
    d = _base()
    d['light'] = {'type': 'point', 'position': [0, 0, 0.5],
                  'intensity': {'type': 'rgb', 'value': [50, 50, 50]}}
    return mi.load_dict(d)


def _mean(scene, spp=128):
    # dr.mean returns a plain Python float in the scalar variants and a 1-element array in
    # the JIT ones; go through numpy so the test reads the same number either way.
    import numpy as np
    return float(np.asarray(mi.render(scene, spp=spp)).mean())


def test01_default_is_all(variant_scalar_rgb):
    """A shape that mentions none of the properties is visible to everything."""
    shape = mi.load_dict({'type': 'rectangle'})
    assert shape.visibility_mask() == int(mi.RayMask.All)


@pytest.mark.parametrize('prop,bit', [
    ('visible_camera', 'Camera'),
    ('visible_diffuse', 'Diffuse'),
    ('visible_glossy', 'Glossy'),
    ('visible_transmission', 'Transmission'),
    ('visible_volume_scatter', 'VolumeScatter'),
    ('visible_shadow', 'Shadow'),
])
def test02_each_property_clears_exactly_its_bit(variant_scalar_rgb, prop, bit):
    shape = mi.load_dict({'type': 'rectangle', prop: False})
    expected = int(mi.RayMask.All) & ~int(getattr(mi.RayMask, bit))
    assert shape.visibility_mask() == expected


def test03_shadow_visibility_lets_nee_through(variants_all_rgb):
    blocked = _mean(_shadow_scene({}))
    passed  = _mean(_shadow_scene({'visible_shadow': False}))

    assert blocked < 1e-4, blocked
    assert passed > 1.0, passed


def test04_shadow_switch_does_not_move_the_others(variants_all_rgb):
    """Turning off a DIFFERENT switch must not open the shadow path."""
    blocked = _mean(_shadow_scene({}))
    for prop in ('visible_camera', 'visible_diffuse', 'visible_glossy',
                 'visible_transmission', 'visible_volume_scatter'):
        other = _mean(_shadow_scene({prop: False}))
        assert other < 1e-4, (prop, other)
        assert dr.allclose(other, blocked, atol=1e-6), (prop, other, blocked)


def test05_camera_visibility_hides_only_primary_rays(variants_all_rgb):
    opaque      = _mean(_camera_scene({}))
    see_through = _mean(_camera_scene({'visible_camera': False}), spp=512)
    absent      = _mean(_camera_scene_without_occluder(), spp=512)

    # The slab is black and its lit side faces away, so with it visible the frame is black.
    assert opaque < 1e-4, opaque
    # And with the flag cleared the camera must see what it sees with no slab at ALL -- not
    # merely more than before, which would also pass for a flag that made the slab grey.
    #
    # The two are not bit-identical and should not be asserted to be: the slab is still THERE
    # for scattered rays, so a path that bounces off the floor into it samples a BSDF and
    # consumes random numbers the occluder-free scene never draws. Both arms are the same
    # unbiased estimator on streams that diverge after the first bounce, so the agreement is
    # Monte-Carlo agreement and the tolerance is a Monte-Carlo tolerance.
    assert absent > 0.05, absent
    assert see_through == pytest.approx(absent, rel=5e-3), (see_through, absent)


def test06_declaring_an_unused_type_changes_nothing(variants_all_rgb):
    """A cleared bit no traced ray carries must not move the image.

    `visible_volume_scatter` is not a ray type any surface integrator traces, so clearing it
    changes the shape's mask without changing the result of any AND the tracer performs. The
    two arms must then produce the same image -- this is what rules out the mask plumbing
    (BLAS bucketing by mask, per-instance masks, the per-lane ray masks) quietly changing
    what gets intersected.
    """
    ref    = _mean(_camera_scene({}))
    walked = _mean(_camera_scene({'visible_volume_scatter': False}))
    assert dr.allclose(ref, walked, atol=1e-6), (ref, walked)

    ref    = _mean(_shadow_scene({'visible_shadow': False}))
    walked = _mean(_shadow_scene({'visible_shadow': False, 'visible_volume_scatter': False}))
    assert dr.allclose(ref, walked, rtol=1e-5), (ref, walked)
