"""Blender's per-lobe bounce budgets on the `path` integrator.

Cycles does not have one bounce limit, it has five, and the counting rule is not the obvious
one. These tests pin the two halves that are easy to get wrong and impossible to notice:

* the ladder equivalence -- in an all-diffuse scene `diffuse_max_depth` truncates the SAME
  walk that `max_depth` truncates, so the two must agree exactly rather than merely closely;
* the partition -- a budget may only ever be tripped by a bounce of its own kind, so a glossy
  or transmission budget must be INERT in a scene that has no such lobe. That one is a
  regression test for a real defect: comparing the counters unconditionally reads
  `glossy_depth >= 0` as exhausted before any glossy bounce has happened, and
  `glossy_max_depth = 0` killed every path at its first vertex.

The tolerance is not a Monte-Carlo tolerance. The two arms of the ladder terminate at the same
vertex and therefore draw the same random numbers, so they are the same computation; measured
run-to-run nondeterminism of the renderer itself (block accumulation order) is ~5e-8 per pixel.
"""

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi

RES = 48
SPP = 128
# Far above the renderer's own run-to-run spread, far below the ~1% gap between ladder rungs.
RTOL = 1e-5


def _render(dielectric=False, **integrator):
    d = mi.cornell_box()
    d['integrator'] = dict(type='path', **integrator)
    d['sensor']['film']['width'] = RES
    d['sensor']['film']['height'] = RES
    d['sensor']['sampler']['sample_count'] = SPP
    if dielectric:
        # A refracting sphere, so the scene has transmission events at all. Everything else in
        # the Cornell box is diffuse.
        d['glass_ball'] = {
            'type': 'sphere',
            'to_world': mi.ScalarTransform4f().translate([0.0, -0.5, 0.2]).scale(0.25),
            'bsdf': {'type': 'dielectric'},
        }
    return float(np.asarray(mi.render(mi.load_dict(d), spp=SPP)).mean())


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test01_diffuse_budget_reproduces_the_depth_ladder(variant_scalar_rgb, k):
    """In an all-diffuse scene the two truncations are the same truncation.

    `max_depth` counts path VERTICES and `diffuse_max_depth` counts diffuse BOUNCES, so the
    correspondence is off by one. If it were off by two, the budget would be spending an extra
    bounce -- which is exactly the error that made the previous `transparent_max_depth`
    workaround render 20% bright.
    """
    assert _render(max_depth=-1, diffuse_max_depth=k) == \
        pytest.approx(_render(max_depth=k + 1), rel=RTOL)


def test02_zero_and_one_are_the_same_budget(variant_scalar_rgb):
    """Cycles compares each count AFTER incrementing it (`bounce >= max`), so a budget of 0
    and a budget of 1 both permit exactly one bounce of that kind. This is surprising enough
    to be worth pinning: it is not an off-by-one, it is the upstream rule."""
    assert _render(max_depth=-1, diffuse_max_depth=0) == \
        pytest.approx(_render(max_depth=-1, diffuse_max_depth=1), rel=RTOL)


@pytest.mark.parametrize("budget", ["glossy_max_depth", "transmission_max_depth"])
def test03_a_budget_is_inert_without_a_bounce_of_its_kind(variant_scalar_rgb, budget):
    """The strictest possible setting of a budget no lobe in the scene can spend must change
    nothing at all. Zero is the strictest, and zero is where the naive spelling fails."""
    assert _render(max_depth=-1, **{budget: 0}) == \
        pytest.approx(_render(max_depth=-1), rel=RTOL)


def test04_transmission_is_charged_to_transmission(variant_scalar_rgb):
    """Add a refracting sphere and the transmission budget bites -- while a diffuse budget set
    far above anything the scene can reach does not. That is the reflect-vs-transmit partition:
    a transmission increments `transmission_bounce` and nothing else."""
    free = _render(dielectric=True, max_depth=-1)
    cut  = _render(dielectric=True, max_depth=-1, transmission_max_depth=0)
    assert cut != pytest.approx(free, rel=1e-3), (cut, free)
    assert cut < free, (cut, free)

    # And the cut is the transmission budget's doing, not a side effect of turning the
    # machinery on: a diffuse budget above the scene's reach leaves the image alone.
    assert _render(dielectric=True, max_depth=-1, diffuse_max_depth=64) == \
        pytest.approx(free, rel=RTOL)


def test05_unset_budgets_change_nothing(variants_all_rgb):
    """The default must be inert in every variant -- this is the guard on every scene that
    does not mention the feature at all."""
    assert _render(max_depth=6, transparent_max_depth=-1, diffuse_max_depth=-1,
                   glossy_max_depth=-1, transmission_max_depth=-1) == \
        pytest.approx(_render(max_depth=6), rel=RTOL)
