"""Cycles' per-sample radiance clamp, which Blender ships ON by default.

Four things have to hold, and each is a separate way to be wrong:

  1. The clamp is INERT at its default, so nothing that does not ask for it changes.
  2. The limit applies to the L1 SUM of the channels and scales the whole spectrum by one
     factor -- so a clamped sample loses brightness and keeps its HUE. A per-channel clamp
     would pass a brightness test and quietly recolour the image.
  3. The stored limit is the UI value TIMES THREE (`scene/integrator.cpp`), so a value of
     10 clamps an L1 sum of 30, not of 10. Off by that factor the image is still plausible.
  4. Direct and indirect are DIFFERENT budgets. A scene with no indirect light must be
     untouched by `clamp_indirect`, and that is the only test here that would catch the two
     accumulation sites having been given the same bounce index.

Reference: `film_clamp_light` in `intern/cycles/kernel/film/light_passes.h`, and the
FLT_MAX-on-zero substitution in `intern/cycles/scene/integrator.cpp`.
"""

import numpy as np
import pytest
import drjit as dr
import mitsuba as mi

RADIANCE = [30.0, 20.0, 10.0]      # L1 sum = 60
L1 = sum(RADIANCE)


def _lookat_emitter_scene(max_depth=2, **integrator):
    """The camera stares at an area light that fills the frame, with a diffuse floor
    below it so the scene also has an indirect term to clamp separately."""
    return mi.load_dict({
        'type': 'scene',
        'integrator': dict(type='path', max_depth=max_depth, **integrator),
        'sensor': {
            'type': 'perspective', 'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 16, 'height': 16,
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': 64, 'seed': 3},
        },
        'lamp': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale(4),
            'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': RADIANCE}},
        },
    })


def _img(scene):
    return np.asarray(mi.render(scene)).astype(np.float64)


def test01_default_is_inert(variants_all_rgb):
    a = _img(_lookat_emitter_scene())
    b = _img(_lookat_emitter_scene(clamp_direct=0.0, clamp_indirect=0.0))
    assert np.allclose(a, b), (a.mean(), b.mean())


def test02_the_limit_is_the_ui_value_times_three(variants_all_rgb):
    """A directly visible emitter is a single unclamped contribution of known size, so the
    scale factor is predictable in closed form rather than merely 'smaller'."""
    unclamped = _img(_lookat_emitter_scene())
    px = unclamped[8, 8]
    assert px.sum() == pytest.approx(L1, rel=1e-4), px

    for ui in (5.0, 10.0, 15.0):
        limit = 3.0 * ui                      # Cycles' own factor of three
        expect = np.array(RADIANCE) * (limit / L1 if L1 > limit else 1.0)
        got = _img(_lookat_emitter_scene(clamp_direct=ui))[8, 8]
        assert np.allclose(got, expect, rtol=1e-4), (ui, got, expect)


def test03_clamping_preserves_hue(variants_all_rgb):
    """One common factor on the whole spectrum -- not a per-channel clip."""
    px = _img(_lookat_emitter_scene(clamp_direct=5.0))[8, 8]
    ref = np.array(RADIANCE)
    assert np.allclose(px / px.sum(), ref / ref.sum(), atol=1e-6), px


def test04_a_limit_above_the_contribution_does_nothing(variants_all_rgb):
    """The clamp is a maximum, not a normalisation: 3 * 25 = 75 > 60, so nothing moves."""
    a = _img(_lookat_emitter_scene())
    b = _img(_lookat_emitter_scene(clamp_direct=25.0))
    assert np.allclose(a, b), (a[8, 8], b[8, 8])


def test05_indirect_budget_does_not_touch_a_direct_only_render(variants_all_rgb):
    """THE test for the two bounce indices. `max_depth=2` admits only single-scattering
    paths, every one of which Cycles calls DIRECT, so an indirect budget -- however
    small -- must change nothing. If the emission site and the NEE site were given the
    same index, one of them would leak into this render."""
    a = _img(_lookat_emitter_scene(max_depth=2))
    b = _img(_lookat_emitter_scene(max_depth=2, clamp_indirect=1e-3))
    assert np.allclose(a, b), (a.mean(), b.mean())


def _cornell(max_depth=8, **kw):
    """A Cornell box: several diffuse surfaces facing each other, so there is genuinely
    multi-scattered light to clamp.

    The first version of test06 used ONE planar wall and a hidden lamp, reasoning that the
    camera then sees only bounced light. That reasoning was wrong in exactly the way this
    file is about: lamp -> wall -> camera is a SINGLE scattering event, which Cycles calls
    DIRECT, and a lone plane cannot bounce light back onto itself. The scene had no
    indirect term at all, so the control failed while the code was right. A control that
    fails for the wrong reason is worth more than one that never runs.
    """
    d = mi.cornell_box()
    d['integrator'] = dict(type='path', max_depth=max_depth, **kw)
    d['sensor']['film']['width'] = 32
    d['sensor']['film']['height'] = 32
    d['sensor']['film']['rfilter'] = {'type': 'box'}
    d['sensor']['sampler']['sample_count'] = 256
    return mi.load_dict(d)


def test06_indirect_budget_does_bite_when_there_is_indirect_light(variants_all_rgb):
    """The positive control for test05: the same budget must be detectable where indirect
    light exists, or test05 passes because the parameter does nothing at all."""
    base = _img(_cornell())
    clamped = _img(_cornell(clamp_indirect=1e-3))
    assert base.mean() > 1e-3, base.mean()
    assert clamped.mean() < base.mean() * 0.95, (base.mean(), clamped.mean())


def test07_the_two_budgets_are_not_the_same_budget(variants_all_rgb):
    """Direct and indirect must be separately reachable on ONE scene. Clamping direct at
    the same tiny value must remove strictly more than clamping indirect does, since every
    indirect path also had to arrive somewhere first."""
    base = _img(_cornell()).mean()
    only_indirect = _img(_cornell(clamp_indirect=1e-3)).mean()
    only_direct = _img(_cornell(clamp_direct=1e-3)).mean()
    assert only_indirect < base, (base, only_indirect)
    assert only_direct < only_indirect, (only_direct, only_indirect)


def test08_a_negative_limit_is_refused(variants_all_rgb):
    for kw in ({'clamp_direct': -1.0}, {'clamp_indirect': -0.5}):
        with pytest.raises(RuntimeError):
            _lookat_emitter_scene(**kw)
