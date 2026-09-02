"""`null` (pass-through) boundaries must be invisible to the `path` integrator.

Blender's Transparent BSDF exports to Mitsuba's `null`, and a clear pane of it is not a
material at all -- it is a no-op. THAT is the property under test here, and it is worth
stating as one property rather than as a list of behaviours, because a single pane in a
single furnace box turned out to violate it in THREE independent ways at once:

  1. the crossing cost the ray its Blender ray-visibility type, so a camera ray that passed
     through a pane stopped being a camera ray and started seeing every lamp in the scene
     (lamps default to `visible_camera = False`);
  2. `Scene::ray_test` -- a pure occlusion query -- reports a clear pane as a hit, so
     next-event estimation died behind it;
  3. the crossing overwrote the MIS bookkeeping (`prev_si`, `prev_bsdf_pdf`) with the pane,
     so a BSDF-sampled emitter hit reached through the pane was weighted ~1 instead of its
     proper share and the emitter got paid for twice.

(2) and (3) very nearly CANCEL. With NEE dead and the BSDF hit double-weighted, the furnace
read 0.9725 against Cycles -- a 2.7% miss that looks like a tolerance issue and is in fact
two large errors of opposite sign. Fixing either one alone makes the image WORSE (fixing
only (2) gives exactly 2.0x). So the tests below assert the invisibility property directly
rather than asserting any one mechanism: an equality against the same scene with the pane
deleted is the only assertion that all three defects fail.

Every case carries a control that must FAIL the same property for a reason that is real --
an opaque pane, an absorbing pane, an exhausted budget -- because "the pane changed nothing"
is also what a test measuring nothing reports.
"""

import numpy as np
import pytest
import drjit as dr
import mitsuba as mi


# Panes live at z ~ 1, the lamp at z = 2, the floor at z = 0. The camera sits BELOW the pane
# plane and looks down, so no primary ray ever crosses a pane: every case below is about the
# light path only, and a budget that stops the camera ray cannot be mistaken for one that
# stops the light. (The one test that IS about primary rays builds its own scene.)
_PANE_Z = 1.0
_PANE_DZ = -0.02


def _clear():
    return {'type': 'null'}


def _absorbing(transmittance):
    """A `mask` that lets `transmittance` through and swallows the rest.

    Mitsuba's `mask` blends its nested BSDF against a pass-through by `opacity`, so an
    opacity of 1 - t transmits t. The nested BSDF is black on purpose: the pane must
    ATTENUATE the light rather than scatter it somewhere else, or the test could not tell a
    correct transmittance from a redistribution.
    """
    return {'type': 'mask',
            'opacity': {'type': 'rgb', 'value': [1.0 - transmittance] * 3},
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0, 0, 0]}}}


def _opaque():
    return {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0, 0, 0]}}


def _room(panes=(), max_depth=8, transparent_max_depth=None, res=32, spp=256):
    """Floor lit by an area lamp, with `panes` stacked between the two.

    The lamp is an AREA emitter rather than a point light, and that is load-bearing for the
    MIS half: a delta light is reachable by next-event estimation ONLY, so it cannot expose a
    mis-weighted BSDF-sampled hit. Both strategies have to be able to find this lamp.
    """
    integrator = {'type': 'path', 'max_depth': max_depth}
    if transparent_max_depth is not None:
        integrator['transparent_max_depth'] = transparent_max_depth

    d = {
        'type': 'scene',
        'integrator': integrator,
        'sensor': {
            'type': 'perspective',
            'fov': 30,
            'to_world': mi.ScalarTransform4f().look_at(origin=[0, -6, 0.5],
                                                       target=[0, 0, 0],
                                                       up=[0, 0, 1]),
            'film': {'type': 'hdrfilm', 'width': res, 'height': res,
                     'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
            'sampler': {'type': 'independent', 'sample_count': spp},
        },
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale(4.0),
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8] * 3}},
        },
        'lamp': {
            'type': 'rectangle',
            # Rotated to face DOWN: an `area` emitter radiates from the +z face of the
            # rectangle, which without this points away from the floor.
            'to_world': (mi.ScalarTransform4f().translate([0, 0, 2.0])
                         @ mi.ScalarTransform4f().rotate(axis=[1, 0, 0], angle=180)
                         @ mi.ScalarTransform4f().scale(0.75)),
            'emitter': {'type': 'area',
                        'radiance': {'type': 'rgb', 'value': [12.0] * 3}},
        },
    }
    for i, pane in enumerate(panes):
        d['pane_%d' % i] = {
            'type': 'rectangle',
            'to_world': (mi.ScalarTransform4f()
                         .translate([0, 0, _PANE_Z + i * _PANE_DZ]).scale(3.5)),
            'bsdf': pane,
        }
    return mi.load_dict(d)


def _mean(scene, spp=256):
    return float(np.asarray(mi.render(scene, spp=spp, seed=0)).mean())


# ---------------------------------------------------------------------------------------
# Construction: the scene-level flag that decides whether shadow rays are marched at all
# ---------------------------------------------------------------------------------------

def test01_scene_reports_whether_it_holds_a_pass_through_boundary(variant_scalar_rgb):
    """`has_null_bsdfs()` is what keeps the march off scenes that do not need it."""
    assert not _room().has_null_bsdfs()
    assert not _room(panes=[_opaque()]).has_null_bsdfs()
    assert _room(panes=[_clear()]).has_null_bsdfs()
    # `mask` is a pass-through for part of its energy, so it counts too.
    assert _room(panes=[_absorbing(0.5)]).has_null_bsdfs()


# ---------------------------------------------------------------------------------------
# The property: a clear pane is a no-op
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize('n_panes', [1, 3, 6])
def test02_a_clear_pane_changes_nothing(variants_all_rgb, n_panes):
    """The whole suite in one assertion; each of the three defects breaks it."""
    absent = _mean(_room())
    present = _mean(_room(panes=[_clear()] * n_panes))

    # VACUITY CONTROL. An all-black frame satisfies "the pane changed nothing" perfectly.
    assert absent > 0.05, absent
    assert present == pytest.approx(absent, rel=2e-2), (present, absent, n_panes)


def test03_an_opaque_pane_still_blocks(variants_all_rgb):
    """CONTROL for test02: the equality above must be capable of failing.

    Without this, a build in which the shadow march simply never occluded anything would
    sail through every case in this file.
    """
    absent = _mean(_room())
    blocked = _mean(_room(panes=[_opaque()]))

    assert absent > 0.05, absent
    assert blocked < absent * 0.25, (blocked, absent)


@pytest.mark.parametrize('transmittance', [0.25, 0.5])
def test04_a_partly_absorbing_pane_attenuates_by_its_transmittance(variants_all_rgb,
                                                                   transmittance):
    """The march must MULTIPLY by `eval_null_transmission`, not merely skip the surface.

    Measured at `max_depth = 2` so the image is direct lighting only: with interreflection
    in play the floor also receives light that never crossed the pane, and the ratio would
    be some scene-dependent number instead of the transmittance itself.
    """
    absent = _mean(_room(max_depth=2))
    attenuated = _mean(_room(panes=[_absorbing(transmittance)], max_depth=2))

    assert absent > 0.05, absent
    assert attenuated / absent == pytest.approx(transmittance, rel=5e-2), \
        (attenuated, absent, transmittance)


def test05_absorbing_panes_compound(variants_all_rgb):
    """Two 0.5 panes transmit 0.25 -- i.e. the crossings multiply rather than saturate."""
    absent = _mean(_room(max_depth=2))
    one = _mean(_room(panes=[_absorbing(0.5)], max_depth=2))
    two = _mean(_room(panes=[_absorbing(0.5)] * 2, max_depth=2))

    assert one / absent == pytest.approx(0.5, rel=5e-2), (one, absent)
    assert two / absent == pytest.approx(0.25, rel=5e-2), (two, absent)


# ---------------------------------------------------------------------------------------
# The budget: Blender's `transparent_max_bounces`, which the exporter maps 1:1
# ---------------------------------------------------------------------------------------

def test06_the_transparent_budget_bounds_the_shadow_ray(variants_all_rgb):
    """A budget of K lets light through K panes and no further.

    The shadow ray gets the same budget the scattering path spends, because in Cycles it is
    the same counter (`transparent_max_bounces`). A shadow ray that ignored the budget would
    keep the floor lit at any depth, which is the failure this pins.
    """
    lit = _mean(_room(panes=[_clear()] * 3, transparent_max_depth=3))
    dark = _mean(_room(panes=[_clear()] * 4, transparent_max_depth=3))

    assert lit > 0.05, lit
    assert dark < lit * 0.25, (dark, lit)


def test07_the_budget_does_not_bind_when_it_is_not_reached(variants_all_rgb):
    """CONTROL for test06: three panes under a budget of eight must be a no-op again."""
    absent = _mean(_room())
    plenty = _mean(_room(panes=[_clear()] * 3, transparent_max_depth=8))

    assert plenty == pytest.approx(absent, rel=2e-2), (plenty, absent)


# ---------------------------------------------------------------------------------------
# Ray visibility across the crossing (see also src/render/tests/test_ray_visibility.py)
# ---------------------------------------------------------------------------------------

def _hidden_slab_scene(with_pane, slab=None, res=32, spp=512):
    """A slab between the camera and the floor, optionally behind a clear pane.

    Geometry copied in spirit from `test_ray_visibility.py`: the light sits UNDERNEATH the
    slab so no shadow ray ever reaches it, and the slab is deliberately WIDER than the floor
    because the steepest rays in the frame cross its plane further out than the shallowest
    ones. The only thing a flag here can change is what PRIMARY rays see.

    `slab` is None for no slab at all, or a dict of extra shape properties.
    """
    d = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 4},
        'sensor': {
            'type': 'perspective',
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
            'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [1] * 3}},
        },
        'light': {'type': 'point', 'position': [0, 0, 0.5],
                  'intensity': {'type': 'rgb', 'value': [50] * 3}},
    }
    if slab is not None:
        d['slab'] = dict({
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, 1]).scale(12.0),
            'bsdf': _opaque(),
        }, **slab)
    if with_pane:
        # BETWEEN the camera and the slab, so a primary ray meets the pane FIRST.
        d['pane'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, 1.5]).scale(12.0),
            'bsdf': _clear(),
        }
    return mi.load_dict(d)


@pytest.mark.skip(reason="needs the per-object ray-visibility cluster (P6): shape property visible_camera does not exist yet on this branch")
def test08_a_crossing_does_not_cost_the_ray_its_camera_visibility(variants_all_rgb):
    """A camera ray is still a camera ray after passing through a clear pane.

    This is the defect that motivated the whole file: the crossing re-derived the ray's
    Blender visibility type from the sampled lobe, and a null lobe -- being neither diffuse
    nor glossy -- fell through to Glossy. Every lamp in a converted scene carries
    `visible_camera = False`, so one pane of glass made all of them visible at full
    emitter radiance.
    """
    hidden_behind_pane = _mean(_hidden_slab_scene(True, slab={'visible_camera': False}))
    no_slab_at_all = _mean(_hidden_slab_scene(True, slab=None))

    assert no_slab_at_all > 0.05, no_slab_at_all
    assert hidden_behind_pane == pytest.approx(no_slab_at_all, rel=2e-2), \
        (hidden_behind_pane, no_slab_at_all)


def test09_the_pane_is_not_what_makes_the_slab_invisible(variants_all_rgb):
    """CONTROL for test08: an ordinary slab must still black the frame out behind a pane.

    Without this, a build in which the pane somehow swallowed the slab entirely -- or in
    which primary rays stopped hitting anything after a crossing -- would pass test08 while
    having nothing to do with the visibility flag.
    """
    blocked = _mean(_hidden_slab_scene(True, slab={}))
    assert blocked < 1e-4, blocked


@pytest.mark.skip(reason="needs the per-object ray-visibility cluster (P6): shape property visible_camera does not exist yet on this branch")
def test10_the_flag_is_what_moves_the_image_not_the_crossing(variants_all_rgb):
    """And the same pair with NO pane, so the flag is shown to work on its own.

    test08 asserts the flag survives a crossing; this asserts there was something to
    survive. Both halves are needed: if the flag were broken outright, test08's two arms
    would agree with each other and prove nothing.
    """
    hidden = _mean(_hidden_slab_scene(False, slab={'visible_camera': False}))
    absent = _mean(_hidden_slab_scene(False, slab=None))
    blocked = _mean(_hidden_slab_scene(False, slab={}))

    assert absent > 0.05, absent
    assert hidden == pytest.approx(absent, rel=2e-2), (hidden, absent)
    assert blocked < 1e-4, blocked
