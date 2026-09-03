"""Consistent path-space regularization in the `path` integrator.

Written against the PAPERS' OWN RULES, the way test_filter_glossy.py is written against
Cycles', because the whole point of the feature is to implement a published consistency
result rather than a plausible-looking blur:

  * Kaplanyan & Dachsbacher, "Path Space Regularization for Holistic and Robust Light
    Transport" (EG / CGF 32(2), 2013), Sect. 5.1 / Eq. 6: the mollified estimator is
    consistent iff the bandwidth shrinks with the sample index n inside
    O(n^{-1/d}) < r_n < O(1); practical sequence r_n = r0 * n^{-lambda},
    lambda in (0; 1/d), lambda = 1/6 for the single-vertex specular (d = 2) case.
  * Weier, Droske, Hanika, Weidlich & Vorba, "Optimised Path Space Regularisation"
    (EGSR / CGF 40(4), 2021), Sect. 4.2: the same sequence applied in ROUGHNESS space
    (h_n = h_0 * n^{-lambda}), which is the form the implementation uses, via the
    `si.min_alpha` floor that filter glossy already plumbs into every microfacet lobe.

The three claims worth dedicated instruments, in rising order of subtlety:

  * THE FLOOR ACTUALLY DECAYS WITH THE SAMPLE INDEX, at exactly r0 * n^{-lambda}
    (test02, a recording-BSDF probe of the mechanism -- images cannot resolve
    per-sample floors, so the probe reads `si.min_alpha` off the interaction itself);
  * THE BIAS VANISHES WITH SPP while blur_glossy's does not (test07 -- the entire
    reason this mode exists; measured, with the noise floor measured beside it);
  * A NULL CROSSING DOES NOT COUNT AS "SEEN INDIRECTLY" (test06): without a separate
    transparent budget a pass-through increments `depth`, so a depth-based guard would
    regularize everything behind a sheet of transparency. The guard must key off the
    first REAL scattering event (`min_ray_pdf` leaving infinity).
"""

import math

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi

# The initial roughness floor r0 used throughout. Big on purpose: the statistical tests
# need the regularized and unregularized estimators to be separable above seed noise.
R0 = 0.4
# The paper's own canonical decay exponent for single-vertex 2D specular mollification.
LAMBDA = 1.0 / 6.0

# See test_filter_glossy.py's PLATE_ALPHA for why 0.1 samples cleanly. The regularization
# tests also use a NEAR-DELTA 0.01 plate where the firefly regime is the point.
PLATE_ALPHA = 0.1
SPP_STAT = 512


def _integrator(blur=None, reg=None, decay=None, max_depth=8):
    d = {'type': 'path', 'max_depth': max_depth}
    if blur is not None:
        d['blur_glossy'] = blur
    if reg is not None:
        d['regularize_roughness'] = reg
    if decay is not None:
        d['regularize_decay'] = decay
    return d


def _glossy_scene(spp=64, seed=0, plate_alpha=PLATE_ALPHA, null_plane=False, **ikw):
    """A glossy plate seen ONLY via a diffuse bounce; lit by a spot aimed at the plate.

    Same construction as test_filter_glossy._glossy_scene and for the same reasons
    (documented there): the camera looks at a diffuse floor, the plate is shaded at
    depth >= 1 after a real scattering event, and the spot keeps the floor's brightest
    pixels BE the reflected footprint rather than direct lighting.
    """
    scene = {
        'type': 'scene',
        'integrator': _integrator(**ikw),
        'sensor': {
            'type': 'perspective',
            'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 1.5, 5], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 48, 'height': 48,
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': spp, 'seed': seed},
        },
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, -1, 0])
                        @ mi.ScalarTransform4f().rotate([1, 0, 0], -90)
                        @ mi.ScalarTransform4f().scale(4),
            'bsdf': {'type': 'diffuse', 'reflectance': 0.8},
        },
        'plate': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 1.2, -1.5])
                        @ mi.ScalarTransform4f().rotate([1, 0, 0], -35)
                        @ mi.ScalarTransform4f().scale(1.2),
            'bsdf': {'type': 'roughconductor', 'alpha': plate_alpha,
                     'distribution': 'ggx'},
        },
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
        scene['nullsheet'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, 3.0])
                        @ mi.ScalarTransform4f().scale(3),
            'bsdf': {'type': 'null'},
        }
    return mi.load_dict(scene)


def _direct_scene(plate_alpha=0.005, null_plane=False, spp=32, **ikw):
    """A near-delta glossy plate seen DIRECTLY (optionally through a null pane).

    The first real scattering vertex must never be regularized, and a null crossing
    must not count as one. Same construction as test_filter_glossy's test05 scene.
    """
    scene = {
        'type': 'scene',
        'integrator': _integrator(max_depth=3, **ikw),
        'sensor': {
            'type': 'perspective', 'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 32, 'height': 32,
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': spp, 'seed': 0},
        },
        'plate': {
            'type': 'rectangle',
            'bsdf': {'type': 'roughconductor', 'alpha': plate_alpha,
                     'distribution': 'ggx'},
        },
        'light': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 2, 2])
                        @ mi.ScalarTransform4f().rotate([1, 0, 0], 90)
                        @ mi.ScalarTransform4f().scale(0.5),
            'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': 50.0}},
        },
    }
    if null_plane:
        scene['nullsheet'] = {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, 2.5])
                        @ mi.ScalarTransform4f().scale(3),
            'bsdf': {'type': 'null'},
        }
    return mi.load_dict(scene)


def _render(scene, spp=64):
    return mi.TensorXf(mi.render(scene, spp=spp))


def _concentration(img):
    """Share of total energy in the brightest 1% of samples; see test_filter_glossy."""
    a = np.asarray(img, dtype=np.float64).ravel()
    total = float(a.sum())
    assert total > 0.0, "the render is black -- the scene is broken, not the feature"
    k = max(1, a.size // 100)
    return float(np.sort(a)[-k:].sum()) / total


# ------------------------------------------------------------------------------------
# The mechanism probe: a BSDF that records the `min_alpha` the integrator hands it.
# ------------------------------------------------------------------------------------

_PROBE_LOG = []


def _make_probe_class():
    class MinAlphaProbe(mi.BSDF):
        """Diffuse-ish BSDF whose only real job is to log `si.min_alpha` per sample().

        The decay law is a PER-SAMPLE property; no image statistic can resolve it,
        because a render averages over all sample indices. The floor travels on the
        interaction, so the shade point itself is the one place it can be read.
        """
        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
            self.m_components = [flags]
            self.m_flags = flags

        def sample(self, ctx, si, sample1, sample2, active=True):
            _PROBE_LOG.append(float(si.min_alpha))
            bs = mi.BSDFSample3f(mi.warp.square_to_cosine_hemisphere(sample2))
            bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
            bs.eta = 1.0
            bs.sampled_type = +mi.BSDFFlags.DiffuseReflection
            bs.sampled_component = 0
            ok = (mi.Frame3f.cos_theta(si.wi) > 0) & (bs.pdf > 0)
            return bs, dr.select(ok, mi.Color3f(0.8), mi.Color3f(0.0))

        def eval(self, ctx, si, wo, active=True):
            return mi.Color3f(0.0)

        def pdf(self, ctx, si, wo, active=True):
            return 0.0

        def eval_pdf(self, ctx, si, wo, active=True):
            return mi.Color3f(0.0), 0.0

        def to_string(self):
            return 'MinAlphaProbe[]'

    return MinAlphaProbe


_PROBE_REGISTERED = set()


def _probe_scene(spp, probe_name, **ikw):
    """Two large parallel probe planes; the camera sits between them looking at one.

    Every camera ray hits the near plane at depth 0 (whose recorded floor must be 0),
    and every cosine bounce off it hits the far plane at depth >= 1 (whose recorded
    floor must be this sample's bandwidth) -- so every sample index n in 1..spp is
    guaranteed to be recorded. No emitter: the probe does not need light, only paths.
    """
    # ONE PROBE TYPE PER PLANE, not shared. Two shapes referencing the same BSDF
    # INSTANCE are auto-merged into a single mesh at scene load, and on this tree the
    # merged mesh has an invalid bounding box -- nothing is ever hit and the probe
    # records nothing (reproduced with two plain `diffuse` rectangles sharing a `ref`;
    # distinct instances are unaffected). Distinct registered types guarantee distinct
    # instances without relying on the dict loader's deduplication rules.
    for suffix in ('near', 'far'):
        name = f'{probe_name}_{suffix}'
        if name not in _PROBE_REGISTERED:
            cls = _make_probe_class()
            mi.register_bsdf(name, lambda props, cls=cls: cls(props))
            _PROBE_REGISTERED.add(name)
    return mi.load_dict({
        'type': 'scene',
        'integrator': _integrator(max_depth=3, **ikw),
        'sensor': {
            'type': 'perspective', 'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 1], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 8, 'height': 8,
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': spp, 'seed': 0},
        },
        'near': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().scale(50),
            'bsdf': {'type': f'{probe_name}_near'},
        },
        'far': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, 2])
                        @ mi.ScalarTransform4f().rotate([1, 0, 0], 180)
                        @ mi.ScalarTransform4f().scale(50),
            'bsdf': {'type': f'{probe_name}_far'},
        },
    })


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test01_default_is_inert(variant):
    """Omitting the parameter and setting it to 0 must be the SAME renderer.

    Also with a (valid) decay exponent present but the floor at zero: the decay knob
    alone must not switch anything on.
    """
    mi.set_variant(variant)
    a = _render(_glossy_scene())
    b = _render(_glossy_scene(reg=0.0))
    c = _render(_glossy_scene(reg=0.0, decay=0.25))
    assert dr.allclose(a, b, atol=1e-6), \
        "regularize_roughness=0 must reproduce the un-parameterised integrator exactly"
    assert dr.allclose(a, c, atol=1e-6), \
        "regularize_decay with a zero floor must not enable anything"


@pytest.mark.parametrize("lam", [None, 0.3])
def test02_floor_decays_at_exactly_r0_n_to_minus_lambda(variant_scalar_rgb, lam):
    """THE MECHANISM: the recorded floors are exactly {r0 * n^-lambda : n = 1..spp}.

    Asserted as a set equality to float precision, for the default lambda (the paper's
    1/6) and for an explicit 0.3 -- so a hard-coded exponent, an off-by-one in n, or a
    decay that silently never reaches the shade point all fail here, without any image
    statistic in the loop. The zero entries are the depth-0 (camera-visible) plane,
    which must never be floored.
    """
    spp = 4
    lam_val = LAMBDA if lam is None else lam
    tag = 'default' if lam is None else 'lam03'

    _PROBE_LOG.clear()
    mi.render(_probe_scene(spp, f'minalpha_probe_on_{tag}', reg=R0, decay=lam),
              spp=spp)
    vals = np.asarray(_PROBE_LOG, dtype=np.float64)
    assert len(vals) > 0, "the probe recorded nothing -- the scene is broken"

    zeros = vals[vals == 0.0]
    nonzero = np.unique(np.round(vals[vals > 0.0], 7))
    expected = np.round(R0 * np.arange(1, spp + 1, dtype=np.float64) ** -lam_val, 7)

    assert len(zeros) > 0, \
        "no zero floors recorded -- the camera-visible vertex is being regularized"
    assert len(nonzero) == spp and np.allclose(
        np.sort(nonzero)[::-1], expected, rtol=1e-5), (
        "recorded floors %r do not match the decay law r0*n^-lambda %r"
        % (np.sort(nonzero)[::-1].tolist(), expected.tolist()))

    # Negative control for the instrument itself: with the feature off, the probe must
    # see nothing but zeros -- otherwise the nonzero readings above prove nothing.
    _PROBE_LOG.clear()
    mi.render(_probe_scene(spp, f'minalpha_probe_off_{tag}'), spp=spp)
    off_vals = np.asarray(_PROBE_LOG, dtype=np.float64)
    assert len(off_vals) > 0 and np.all(off_vals == 0.0), \
        "min_alpha is nonzero with regularization disabled -- the mode leaks when off"


def test03_composes_with_filter_glossy_by_max(variant_scalar_rgb):
    """With both features on, no shade point may see LESS than the regularization floor.

    blur_glossy computes its own (path-dependent) floor; composition is max, so the
    smallest nonzero floor observable anywhere is bounded below by the smallest
    regularization bandwidth of the run.
    """
    spp = 4
    _PROBE_LOG.clear()
    mi.render(_probe_scene(spp, 'minalpha_probe_both', reg=R0, blur=4.0), spp=spp)
    vals = np.asarray(_PROBE_LOG, dtype=np.float64)
    nonzero = vals[vals > 0.0]
    assert len(nonzero) > 0, "the probe recorded no floored vertices"
    lower = R0 * spp ** -LAMBDA
    assert float(nonzero.min()) >= lower - 1e-5, (
        "a shade point saw floor %.6f, below the smallest regularization bandwidth "
        "%.6f -- blur_glossy is REPLACING the floor instead of composing by max"
        % (float(nonzero.min()), lower))


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test04_blur_glossy_is_untouched(variant):
    """`blur_glossy` semantics must be bit-for-bit independent of the new knobs.

    The Cycles-parity arm is a different feature with a different (fixed-bias)
    contract; adding regularize_roughness=0 beside it must change nothing.
    """
    mi.set_variant(variant)
    a = _render(_glossy_scene(blur=1.0))
    b = _render(_glossy_scene(blur=1.0, reg=0.0))
    assert dr.allclose(a, b, atol=1e-6), \
        "a disabled regularizer changed a blur_glossy render"


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test05_first_vertex_is_never_regularized(variant):
    """A directly visible near-delta surface must be exactly untouched at any spp.

    Includes the POSITIVE control (the broken-guard pattern of test_filter_glossy's
    test06): the image a broken guard would produce -- the plate floored at the
    SMALLEST bandwidth of the run, the least detectable failure -- must be separable
    from the base render, or the equality assertion above it cannot fail.
    """
    mi.set_variant(variant)
    spp = 32
    a = _render(_direct_scene(spp=spp), spp=spp)
    b = _render(_direct_scene(spp=spp, reg=R0), spp=spp)
    assert dr.allclose(a, b, atol=1e-6), \
        "a directly-visible surface was regularized at the first vertex"

    smallest_floor = R0 * spp ** -LAMBDA
    broken = _render(_direct_scene(spp=spp, plate_alpha=smallest_floor), spp=spp)
    d = float(np.mean(np.abs(np.asarray(broken) - np.asarray(a))))
    assert d > 1e-4, (
        "the instrument is blind: flooring the plate at the smallest bandwidth "
        "(%.4f) moved the image only %.2e -- the equality above cannot fail"
        % (smallest_floor, d))


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test06_null_crossing_is_not_indirect_visibility(variant):
    """A pass-through pane must not make what is behind it 'indirectly seen'.

    Without a separate transparent budget a null interaction increments `depth`, so a
    depth-based guard regularizes everything behind a sheet of transparency -- while
    the physics says a pane crossing is not a scattering event (the same rule Cycles
    spells LABEL_TRANSPARENT, and that test_filter_glossy's test06 polices for
    blur_glossy). Same scene both arms, only the flag differs, so the images must be
    identical -- the guard keys off the first REAL scattering event. test05's positive
    control already proved the instrument can see a floored plate.
    """
    mi.set_variant(variant)
    spp = 32
    a = _render(_direct_scene(spp=spp, null_plane=True), spp=spp)
    b = _render(_direct_scene(spp=spp, null_plane=True, reg=R0), spp=spp)
    assert dr.allclose(a, b, atol=1e-6), \
        "a null crossing enabled regularization -- transparency is blurring the scene"


def test07_bias_decays_with_spp_unlike_blur_glossy(variant_cuda_ad_rgb):
    """CONSISTENCY, the whole point: bias -> 0 as spp grows; blur_glossy's does not.

    For each mode and each spp rung, the bias is estimated as the seed-average of
    (mode render - unregularized render) under PAIRED seeds -- the expectation of that
    difference is exactly the systematic error, and pairing cancels most of the common
    variance. The regularized floor averages (1/S)*sum(n^-lambda) ~ S^-lambda over S
    samples, so its bias must FALL along the ladder; blur_glossy's floor never hears
    about n, so its bias must NOT fall materially. The noise floor of the estimator is
    measured (an off-vs-off arm with independent seeds) rather than assumed.

    CUDA-only: the ladder needs hundreds of renders; scalar would take minutes for no
    additional coverage of integrator logic (the megakernel exercises the same code).
    """
    spps = [1, 16, 256]
    n_seeds = 16

    def bias(mode_kw, spp):
        acc = None
        for s in range(n_seeds):
            on = np.asarray(_render(
                _glossy_scene(spp=spp, seed=s, plate_alpha=0.05, **mode_kw), spp=spp),
                dtype=np.float64)
            off = np.asarray(_render(
                _glossy_scene(spp=spp, seed=s, plate_alpha=0.05), spp=spp),
                dtype=np.float64)
            d = on - off
            acc = d if acc is None else acc + d
        return float(np.mean(np.abs(acc / n_seeds)))

    # Noise floor of the same estimator when the TRUE bias is zero: off-vs-off with
    # independent seeds, at the largest rung (where the consistency claim must hold).
    def noise(spp):
        acc = None
        for s in range(n_seeds):
            x = np.asarray(_render(_glossy_scene(spp=spp, seed=s, plate_alpha=0.05),
                                   spp=spp), dtype=np.float64)
            y = np.asarray(_render(_glossy_scene(spp=spp, seed=1000 + s,
                                                 plate_alpha=0.05), spp=spp),
                           dtype=np.float64)
            d = x - y
            acc = d if acc is None else acc + d
        return float(np.mean(np.abs(acc / n_seeds)))

    reg = {s: bias({'reg': R0}, s) for s in spps}
    blur = {s: bias({'blur': 4.0}, s) for s in spps}
    nf = noise(spps[-1])

    # The table is the report; print it so a failure comes with its evidence.
    print("\nbias vs spp (mean |seed-avg difference| to the unregularized estimator):")
    print("  spp   regularize      blur_glossy")
    for s in spps:
        print("  %4d  %.6f       %.6f" % (s, reg[s], blur[s]))
    print("  noise floor at spp=%d (off vs off): %.6f" % (spps[-1], nf))

    # The claim needs signal: the low-spp regularization bias must stand well above
    # the measurement's own zero-bias noise, or none of the ratios below mean anything.
    assert reg[spps[0]] > 5.0 * nf, (
        "the instrument is blind: regularization bias at spp=%d (%.6f) is within the "
        "noise floor (%.6f)" % (spps[0], reg[spps[0]], nf))
    assert blur[spps[-1]] > 5.0 * nf, (
        "the instrument is blind for the control arm: blur_glossy bias (%.6f) is "
        "within the noise floor (%.6f)" % (blur[spps[-1]], nf))

    # Consistency: the regularization bias falls monotonically along the ladder and
    # ends well below where it started. (1/S)*sum(n^-1/6) predicts ~0.48 at S=256;
    # 0.7 leaves room for the nonlinearity of bias-vs-floor and for noise.
    for lo, hi in zip(spps, spps[1:]):
        assert reg[hi] < reg[lo] + 2.0 * nf, (
            "regularization bias ROSE from spp=%d to %d (%.6f -> %.6f)"
            % (lo, hi, reg[lo], reg[hi]))
    assert reg[spps[-1]] < 0.7 * reg[spps[0]], (
        "regularization bias did not decay: %.6f at spp=%d vs %.6f at spp=%d -- the "
        "floor is not shrinking with the sample index"
        % (reg[spps[-1]], spps[-1], reg[spps[0]], spps[0]))

    # The negative control: blur_glossy's bias is FIXED by construction. If it decays
    # the same way, the assertion above measured spp-dependence of the ESTIMATOR, not
    # consistency of the regularizer.
    assert blur[spps[-1]] > 0.7 * blur[spps[0]] - 2.0 * nf, (
        "blur_glossy's bias decayed with spp (%.6f -> %.6f), which it cannot -- the "
        "consistency instrument is measuring something else"
        % (blur[spps[0]], blur[spps[-1]]))


def _glass_lamp_scene(spp, seed, reg=None, decay=None, alpha=0.02):
    """A lamp enclosed in a near-delta rough-dielectric shell, lighting a bare floor.

    This is the literature's own hard case -- Kaplanyan & Dachsbacher's introduction
    cites exactly "a light source ... enclosed in a glass fixture" as where unbiased
    methods fail. NEE from the floor is DEAD (shadow rays stop at the dielectric), so
    every unit of light crosses two near-delta refractions found by BSDF sampling
    alone: rare, huge-weight events -- fireflies. The camera sees ONLY bare floor;
    the fixture is out of frame, because a directly-viewed shell is depth 0 and
    deliberately untouched, which would dilute the measurement with variance the
    feature must not remove.
    """
    integ = {'type': 'path', 'max_depth': 12}
    if reg is not None:
        integ['regularize_roughness'] = reg
    if decay is not None:
        integ['regularize_decay'] = decay
    return mi.load_dict({
        'type': 'scene',
        'integrator': integ,
        'sensor': {
            'type': 'perspective', 'fov': 40,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[2.6, 1.6, 2.6], target=[2.2, -1, 0], up=[0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 48, 'height': 48,
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': spp, 'seed': seed},
        },
        'floor': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, -1, 0])
                        @ mi.ScalarTransform4f().rotate([1, 0, 0], -90)
                        @ mi.ScalarTransform4f().scale(8),
            'bsdf': {'type': 'diffuse', 'reflectance': 0.8},
        },
        'shell': {
            'type': 'sphere', 'center': [0, 0.2, 0], 'radius': 0.45,
            'bsdf': {'type': 'roughdielectric', 'alpha': alpha,
                     'distribution': 'ggx', 'int_ior': 1.5},
        },
        'lamp': {
            'type': 'sphere', 'center': [0, 0.2, 0], 'radius': 0.08,
            'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': 500.0}},
        },
    })


def test08_error_reduction_at_low_spp(variant_cuda_ad_rgb):
    """The trade the feature buys: lower ERROR at low spp on the glass-lamp scene.

    The instrument is MSE against a shared converged reference -- variance plus
    bias squared, which is the accounting both papers use (Weier et al. Fig. 7) --
    NOT per-pixel variance across a handful of seeds. That simpler instrument was
    built first and measured BLIND, twice over, on this feature's own turf: on a
    near-delta plate lit by a delta spot, the unregularized estimator's caustic
    arrives only through alignment spikes so rare that eight seeds all miss them
    identically -- an estimator that has not found the transport yet reads as
    ZERO-variance, so "regularization reduces variance" fails against an arm whose
    measured variance is vacuously tiny. And per unit of transported energy the
    relative noise of the two arms measured EQUAL (var/mean^2 within 2% on the top
    caustic pixels), so raw variance mostly re-measures how much energy each arm
    carries. MSE against the truth charges the unregularized arm for the fireflies
    it has not yet realized, which is the error a user actually sees.

    The reference is computed BOTH ways -- unregularized at high spp, and
    regularized with the fastest allowed decay (lambda=0.4, whose floor drops below
    the shell's alpha after ~180 samples, leaving >97% of its samples running the
    unbiased estimator) -- and the two families must AGREE, or the reference (and
    with it the whole comparison) is broken.
    """
    spp, n_seeds, ref_spp = 16, 8, 8192

    def img(s, **kw):
        return np.asarray(_render(_glass_lamp_scene(spp, s, **kw), spp=spp),
                          dtype=np.float64)

    def ref_img(s, **kw):
        return np.asarray(_render(_glass_lamp_scene(ref_spp, s, **kw), spp=ref_spp),
                          dtype=np.float64)

    ref_off = 0.5 * (ref_img(11) + ref_img(12))
    ref_reg = 0.5 * (ref_img(111, reg=R0, decay=0.4) + ref_img(112, reg=R0, decay=0.4))
    assert abs(ref_off.mean() - ref_reg.mean()) < 0.02 * ref_off.mean(), (
        "the two reference families disagree (%.4f vs %.4f) -- the reference is "
        "broken and no MSE below can be trusted" % (ref_off.mean(), ref_reg.mean()))
    ref = 0.5 * (ref_off + ref_reg)

    mse_off = [float(np.mean((img(s) - ref) ** 2)) for s in range(n_seeds)]
    mse_on = [float(np.mean((img(s, reg=R0) - ref) ** 2)) for s in range(n_seeds)]
    print("\nMSE vs converged reference at spp=%d over %d seeds:" % (spp, n_seeds))
    print("  off: mean %.3e  %r" % (np.mean(mse_off), ['%.2e' % e for e in mse_off]))
    print("  on : mean %.3e  %r" % (np.mean(mse_on), ['%.2e' % e for e in mse_on]))

    assert np.mean(mse_on) < 0.75 * np.mean(mse_off), (
        "regularization did not reduce error at low spp: MSE %.3e -> %.3e"
        % (np.mean(mse_off), np.mean(mse_on)))
    # Not just on average: the WORST regularized seed must beat the average
    # unregularized one, or the mean was carried by luck.
    assert max(mse_on) < np.mean(mse_off), (
        "the worst regularized seed (%.3e) does not beat the mean unregularized "
        "error (%.3e)" % (max(mse_on), np.mean(mse_off)))


@pytest.mark.parametrize("variant", ['scalar_rgb', 'cuda_ad_rgb'])
def test09_invalid_parameters_are_refused(variant):
    """Never silently reinterpret a nonsensical value.

    The decay bounds are the consistency condition itself: lambda = 0 is blur-glossy's
    fixed bias wearing this feature's name, and lambda >= 1/2 violates Eq. 6's lower
    bound for a 2D mollification -- both must be refused, not clamped.
    """
    mi.set_variant(variant)
    with pytest.raises(RuntimeError, match="non-negative"):
        _glossy_scene(reg=-0.1)
    for bad in (0.0, 0.5, -0.2, 1.0):
        with pytest.raises(RuntimeError, match="open interval"):
            _glossy_scene(reg=R0, decay=bad)
