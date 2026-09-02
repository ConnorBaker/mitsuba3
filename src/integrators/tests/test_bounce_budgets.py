"""The `path` integrator REFUSES Cycles' per-lobe bounce budgets, and says why.

This file used to pin the transcription of Blender's `diffuse_bounces` / `glossy_bounces` /
`transmission_bounces` (the ladder equivalence, the after-increment counting rule, the
reflect-vs-transmit partition). Those budgets were REMOVED on 2026-09-02: truncating
transport per lobe is a nonphysical variance control, and the project's posture is to
evaluate Cycles' hacks rather than replicate them -- parity with a Blender scene whose
budgets bind is achieved by raising the scene's per-lobe bounces to `max_bounces`, not by
truncating Mitsuba the same way. The history of the transcription (and its measured
furnace ladder) is in git; the transparent budget SURVIVES, because a null crossing is not
a scattering event at all -- its behaviour is pinned in `test_transparent_boundaries.py`.

What must hold now:

* the constructor REFUSES the removed keys BY NAME -- silently ignoring them would render
  a different image than the scene asked for, and Mitsuba's unqueried-property warning is
  not loud enough to carry that;
* the refusal is TARGETED -- `transparent_max_depth` (kept) must still load; this is the
  negative control that separates "refuses removed budgets" from "refuses everything";
* an explicit `transparent_max_depth = -1` stays bit-inert against not mentioning it.
"""

import pytest
import numpy as np
import mitsuba as mi

RES = 32
SPP = 64
# The two arms are the same computation; measured run-to-run nondeterminism ~5e-8/px.
RTOL = 1e-5

REMOVED = ["diffuse_max_depth", "glossy_max_depth", "transmission_max_depth"]


def _scene_dict(**integrator):
    d = mi.cornell_box()
    d['integrator'] = dict(type='path', **integrator)
    d['sensor']['film']['width'] = RES
    d['sensor']['film']['height'] = RES
    d['sensor']['sampler']['sample_count'] = SPP
    return d


def _render(**integrator):
    return float(np.asarray(
        mi.render(mi.load_dict(_scene_dict(**integrator)), spp=SPP)).mean())


@pytest.mark.parametrize("key", REMOVED)
def test01_removed_budget_keys_are_refused_by_name(variant_scalar_rgb, key):
    """A scene naming a removed budget must fail to LOAD, and the error must name the key
    and point at the remedy -- a reader holding an old exported scene gets the migration
    instruction in the traceback, not a silently different image."""
    with pytest.raises(RuntimeError, match=key):
        mi.load_dict(_scene_dict(max_depth=6, **{key: 4}))


def test02_the_refusal_is_targeted(variant_scalar_rgb):
    """NEGATIVE CONTROL: `transparent_max_depth` is kept, so it must still load and render.
    Without this, test01 is satisfied by a constructor that refuses every unknown key --
    a gate that can only say no proves nothing by saying it."""
    val = _render(max_depth=6, transparent_max_depth=8)
    assert np.isfinite(val) and val > 0.0


def test03_explicit_default_is_inert(variants_all_rgb):
    """`transparent_max_depth = -1` spelled out must equal not mentioning it, in every
    variant -- the guard on every scene that does not use the feature."""
    assert _render(max_depth=6, transparent_max_depth=-1) == \
        pytest.approx(_render(max_depth=6), rel=RTOL)
