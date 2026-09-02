"""Cycles' "filter glossy" (`cycles.blur_glossy`) in the `path` integrator.

Written against CYCLES' OWN RULES, not against the implementation, because the whole point
of the feature is to reproduce another renderer:

  * `scene/integrator.cpp`   -- `filter_glossy = (blur_glossy == 0) ? FLT_MAX : 1/blur_glossy`
  * `kernel/integrator/path_state.h` + `shade_surface.h`
                             -- `min_ray_pdf` starts FLT_MAX, then
                                `min(unguided_bsdf_pdf, min_ray_pdf)` after every
                                NON-transparent BSDF sample
  * `kernel/integrator/surface_shader.h`
                             -- `blur_pdf = filter_glossy * min_ray_pdf`; for `blur_pdf < 1`,
                                `blur_roughness = sqrt(1 - blur_pdf) * 0.5`
  * `kernel/closure/bsdf_microfacet.h` -- `alpha_{x,y} = max(roughness, alpha_{x,y})`
  * `kernel/closure/bsdf.h`  -- diffuse closures fall to `default: break`

Two of those are the ones easy to get wrong in the direction that still produces a
plausible image, so they get dedicated tests: DIFFUSE IS NOT BLURRED (test03) and a
TRANSPARENT/null bounce MUST NOT tighten the bound (test06).
"""

import math

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi

# A plate rough enough to SAMPLE, still far below every blur floor the feature can
# impose. See `_glossy_scene`.
PLATE_ALPHA = 0.1

# Enough samples that the statistic below is decided by the feature rather than by
# the seed; the tests that need it measure their own seed-to-seed noise floor.
SPP_STAT = 512


def _glossy_scene(blur_glossy=None, spp=64, null_plane=False, seed=0,
                  plate_alpha=PLATE_ALPHA):
    """A glossy plate seen ONLY via a diffuse bounce.

    Filter glossy keys off `min_ray_pdf`, which is infinite at the first vertex -- so a
    camera-visible glossy surface is never blurred and a scene that only had one would
    make every test here vacuous. The camera therefore looks at a DIFFUSE floor, and the
    light reaches that floor by reflecting off the glossy plate: the plate is shaded at
    depth >= 1, after a diffuse sample whose pdf is <= 1/pi.
    """
    integrator = {'type': 'path', 'max_depth': 8}
    if blur_glossy is not None:
        integrator['blur_glossy'] = blur_glossy

    scene = {
        'type': 'scene',
        'integrator': integrator,
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 1.5, 5], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 48, 'height': 48,
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': spp, 'seed': seed},
        },
        # Diffuse floor -- what the camera actually looks at.
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, -1, 0])
                        @ mi.ScalarTransform4f().rotate([1, 0, 0], -90)
                        @ mi.ScalarTransform4f().scale(4),
            'bsdf': {'type': 'diffuse', 'reflectance': 0.8},
        },
        # The glossy plate. Its alpha is far below every blur floor the feature can
        # impose, so `max(alpha, min_alpha)` is decided by `min_alpha` whenever it fires.
        # It used to be 0.005, which is the more obvious choice and was WRONG: a
        # near-delta lobe lit by a delta spot is essentially unsamplable, the estimator
        # went firefly-dominated (image mean 5.8e-05; two SEEDS of the SAME scene differed
        # per-pixel by 130% of that mean), and no assertion in this file could mean
        # anything against that noise. At 0.1 the same experiment converges.
        'plate': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 1.2, -1.5])
                        @ mi.ScalarTransform4f().rotate([1, 0, 0], -35)
                        @ mi.ScalarTransform4f().scale(1.2),
            'bsdf': {'type': 'roughconductor', 'alpha': plate_alpha,
                     'distribution': 'ggx'},
        },
        # A SPOT aimed at the plate, not an area light. With an area light the floor is
        # also lit DIRECTLY, and direct floor lighting dominates every whole-image
        # statistic -- so `max(image)` measured the one thing filter glossy cannot touch
        # and the test failed while the feature worked. Narrowing the illumination to the
        # plate makes the floor's brightest pixel BE the reflected footprint, which is
        # exactly the quantity the blur is supposed to spread.
        'light': {
            'type': 'spot',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 1.2, 2.2], target=[0, 1.2, -1.5], up=[0, 1, 0]),
            'cutoff_angle': 9.0,
            'beam_width': 7.0,
            'intensity': {'type': 'rgb', 'value': 900.0},
        },
    }
    if null_plane:
        # A pass-through sheet between the camera and everything else. Cycles guards the
        # `min_ray_pdf` update with `!(label & LABEL_TRANSPARENT)`, so this must change
        # NOTHING. Get that wrong and every transparent surface silently blurs the scene
        # behind it.
        scene['nullsheet'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, 3.0])
                        @ mi.ScalarTransform4f().scale(3),
            'bsdf': {'type': 'null'},
        }
    return mi.load_dict(scene)


def _render(scene, spp=64):
    return mi.TensorXf(mi.render(scene, spp=spp))


def _concentration(img):
    """Share of the image's total energy carried by its brightest 1% of samples.

    This is what filter glossy DOES, written as a number: widening a microfacet lobe moves
    energy out of the near-specular peak into its surroundings, so the peak's share falls.

    Two more obvious statistics were tried first and BOTH are blind here. `max(image)` and
    "fraction of samples above 5% of peak" are each pinned by the directly-lit floor, which
    filter glossy never touches: across a 0 -> 0.5 -> 1 -> 2 -> 4 blur ladder the peak did
    not move in the fifth significant digit (1.1747 at every rung) and the fraction moved
    from 0.0408 to 0.0412. The same ladder moves this statistic 0.4118 -> 0.3633 -> 0.3557
    -> 0.3522 -> 0.3507 against a seed-to-seed noise floor of 0.0015.
    """
    a = np.asarray(img, dtype=np.float64).ravel()
    total = float(a.sum())
    # A black image would satisfy every "concentration fell" assertion in this file for
    # free. It is a broken scene, not a passing test.
    assert total > 0.0, "the render is black -- the scene is broken, not the feature"
    k = max(1, a.size // 100)
    return float(np.sort(a)[-k:].sum()) / total


def _seed_noise(blur, **kw):
    """How much `_concentration` moves when ONLY the RNG seed changes.

    Every threshold below is stated against this rather than against a constant picked to
    make the test pass, because the quantity it has to beat is noise.
    """
    a = _concentration(_render(_glossy_scene(blur_glossy=blur, spp=SPP_STAT, seed=0, **kw),
                               spp=SPP_STAT))
    b = _concentration(_render(_glossy_scene(blur_glossy=blur, spp=SPP_STAT, seed=7, **kw),
                               spp=SPP_STAT))
    return a, abs(b - a)


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test01_default_is_inert(variant):
    """Omitting the parameter and setting it to 0 must be the SAME renderer."""
    mi.set_variant(variant)
    a = _render(_glossy_scene(blur_glossy=None))
    b = _render(_glossy_scene(blur_glossy=0.0))
    assert dr.allclose(a, b, atol=1e-6), \
        "blur_glossy=0 must reproduce the un-parameterised integrator exactly"


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test02_glossy_indirect_is_blurred(variant):
    """The feature must actually DO something on the path it targets.

    Without this the rest of the file could pass with a no-op implementation.
    """
    mi.set_variant(variant)
    off, noise = _seed_noise(0.0)
    on = _concentration(_render(_glossy_scene(blur_glossy=1.0, spp=SPP_STAT),
                                spp=SPP_STAT))
    assert off - on > max(10.0 * noise, 0.02), (
        "blur_glossy=1.0 did not spread the glossy lobe: concentration %.4f -> %.4f "
        "(drop %.4f), against a seed-to-seed noise floor of %.4f" % (off, on, off - on, noise))


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test03_diffuse_is_not_blurred(variant):
    """`bsdf.h` sends diffuse closures to `default: break`.

    A wholly diffuse scene must be bit-for-bit unaffected. This is the correction the
    Cycles source forced: blurring diffuse would be invisible in an image and wrong.
    """
    mi.set_variant(variant)
    def cbox(blur):
        d = mi.cornell_box()
        d['integrator'] = {'type': 'path', 'max_depth': 6, 'blur_glossy': blur}
        d['sensor']['film']['width'] = 48
        d['sensor']['film']['height'] = 48
        d['sensor']['sampler']['sample_count'] = 32
        return mi.load_dict(d)
    a = _render(cbox(0.0), spp=32)
    b = _render(cbox(1.0), spp=32)
    assert dr.allclose(a, b, atol=1e-6), \
        "a diffuse-only scene must be untouched by filter glossy"


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test04_more_blur_glossy_means_more_blur(variant):
    """`filter_glossy = 1/blur_glossy`, so a LARGER UI value blurs MORE.

    An implementation that forgot the inversion would order these backwards.

    The ladder SATURATES at the top and that is required, not slack: `blur_roughness =
    sqrt(1 - blur_pdf) * 0.5` is bounded by 0.5 however small `blur_pdf` gets, so the last
    rungs differ by less than the seed noise. Asserting a STRICT drop at every rung would
    be asserting something the formula forbids.
    """
    mi.set_variant(variant)
    ladder = (0.0, 0.5, 1.0, 2.0, 4.0)
    _, noise = _seed_noise(0.0)
    tol = max(4.0 * noise, 0.005)
    conc = [_concentration(_render(_glossy_scene(blur_glossy=b, spp=SPP_STAT), spp=SPP_STAT))
            for b in ladder]
    for i in range(len(ladder) - 1):
        assert conc[i + 1] <= conc[i] + tol, (
            "concentration ROSE from blur_glossy=%g to %g (%.4f -> %.4f, tolerance %.4f); "
            "larger blur_glossy must blur MORE, so `filter_glossy = 1/blur_glossy` is "
            "likely inverted. Ladder: %r" % (ladder[i], ladder[i + 1], conc[i], conc[i + 1],
                                             tol, conc))
    assert conc[0] - conc[-1] > 0.02, (
        "the whole ladder moved only %.4f -- blur_glossy is barely doing anything: %r"
        % (conc[0] - conc[-1], conc))


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test05_first_vertex_is_never_blurred(variant):
    """`min_ray_pdf` is infinite at a camera-visible surface, so `blur_pdf` is too.

    A scene whose only glossy surface is seen DIRECTLY must be unaffected -- this is what
    keeps sharp reflections sharp in the image plane.
    """
    mi.set_variant(variant)
    def direct(blur):
        return mi.load_dict({
            'type': 'scene',
            'integrator': {'type': 'path', 'max_depth': 3, 'blur_glossy': blur},
            'sensor': {
                'type': 'perspective', 'fov': 45,
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
                'film': {'type': 'hdrfilm', 'width': 32, 'height': 32,
                         'rfilter': {'type': 'box'}},
                'sampler': {'type': 'independent', 'sample_count': 32, 'seed': 0},
            },
            'plate': {
                'type': 'rectangle',
                'bsdf': {'type': 'roughconductor', 'alpha': 0.005},
            },
            'light': {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate([0, 2, 2])
                            @ mi.ScalarTransform4f().rotate([1, 0, 0], 90)
                            @ mi.ScalarTransform4f().scale(0.5),
                'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': 50.0}},
            },
        })
    a = _render(direct(0.0), spp=32)
    b = _render(direct(1.0), spp=32)
    assert dr.allclose(a, b, atol=1e-6), \
        "a directly-visible glossy surface must not be blurred at the first vertex"


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test06_null_bounce_does_not_tighten_the_bound(variant):
    """Cycles guards the update with `!(label & LABEL_TRANSPARENT)`.

    Passing through a `null` BSDF is not a scattering event. If it updated `min_ray_pdf`,
    a sheet of transparency in front of the camera would blur the whole scene behind it --
    which still LOOKS like a render, which is why it needs a test.
    """
    mi.set_variant(variant)

    # `null`'s sample pdf is 1, so the UI value matters to whether this test can fail at
    # all: a BROKEN guard at blur_glossy=1 computes `blur_pdf = 1/1 * 1 = 1.0`, which is
    # not `< 1` and therefore blurs NOTHING. The test would pass on exactly the build it
    # exists to catch. At blur_glossy=4 the same broken guard yields
    # `blur_roughness = sqrt(1 - 0.25) * 0.5 = 0.4330`, which is enormous.
    BLUR = 4.0
    broken_alpha = math.sqrt(1.0 - 1.0 / BLUR) * 0.5

    base, noise = _seed_noise(BLUR)
    sheet = _concentration(_render(
        _glossy_scene(blur_glossy=BLUR, spp=SPP_STAT, null_plane=True), spp=SPP_STAT))

    # POSITIVE CONTROL, and the reason this test is worth having. A broken guard forces
    # the plate to `broken_alpha`, which is simulated exactly by rendering with NO sheet
    # and that alpha. If the statistic cannot separate THAT from the real scene, it cannot
    # detect the bug either and a green result here means nothing.
    broken = _concentration(_render(
        _glossy_scene(blur_glossy=BLUR, spp=SPP_STAT, plate_alpha=broken_alpha),
        spp=SPP_STAT))
    assert abs(broken - base) > 0.008, (
        "the instrument is blind: forcing the plate to the alpha a broken guard would "
        "impose (%.4f) moved concentration only %.4f -- this test cannot fail"
        % (broken_alpha, abs(broken - base)))

    # The sheet may shift the sample sequence; it may not shift the PHYSICS. Its effect
    # must therefore be no larger than simply re-rolling the seed.
    assert abs(sheet - base) <= max(2.0 * noise, 0.008), (
        "the null sheet moved concentration by %.4f (%.4f -> %.4f), beyond the %.4f a seed "
        "change costs -- a transparent bounce is tightening min_ray_pdf. A broken guard "
        "moves it %.4f." % (abs(sheet - base), base, sheet, noise, abs(broken - base)))


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test07_negative_is_refused(variant):
    """Never silently reinterpret a nonsensical value."""
    mi.set_variant(variant)
    with pytest.raises(RuntimeError, match="non-negative"):
        _glossy_scene(blur_glossy=-1.0)
