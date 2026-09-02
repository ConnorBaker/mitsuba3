from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
from ._base import TextureBase

from .blender_hash import hash_float4_to_float

# Blender's Ambient Occlusion node, transcribed from Cycles' `svm_ao`
# (intern/cycles/kernel/svm/ao.h):
#
#   if (inside) N = -N
#   make_orthonormals(N, T, B)
#   for sample in range(num_samples):
#       d = sample_uniform_disk(rand)
#       D = to_global((d.x, d.y, sqrt(1 - d.d)), T, B, N)      # Malley, i.e. cosine-weighted
#       unoccluded += not occluded(P, D, tmax = max_dist)
#   return unoccluded / num_samples
#
# and the two outputs:  AO = that fraction,  Color = AO * the Color input.
#
# THIS IS THE ONE NODE THAT CANNOT BE PIXEL-EXACT AGAINST CYCLES, and saying so is the
# point. Cycles draws its directions from the PATH's stratified RNG, which no texture here
# can see; this draws them from a hash of the shading point. Both are unbiased estimators of
# the same integral and both converge to it, but sample for sample they disagree, so AO
# parity is a DISTRIBUTIONAL claim (converged means, matching within the noise) and never a
# per-pixel one. The per-point hash has one consequence worth stating: a shading point hit
# at exactly the same coordinates every time gets exactly the same estimate every time, so
# extra samples per pixel do not average it down. In a rendered image sub-pixel jitter moves
# `si.p`, so they do.
#
# A texture has no scene, and Mitsuba has no back-pointer from a shape to the scene that
# owns it, so the scene is BOUND after `load_dict` -- see `bind_scene` below. Evaluating an
# unbound instance raises rather than returning 1.0: "everything is unoccluded" is exactly
# the plausible wrong picture that would never be reported.

CHANNELS = ('AO', 'Color')

_PENDING = []


def bind_scene(scene):
    """Hand every AmbientOcclusion texture built since the last call the scene it traces
    against, and return how many were bound.

    Call this immediately after `mi.load_dict` / `mi.load_file`. The list is cleared, so a
    second scene loaded afterwards binds only its own textures.
    """
    n = 0
    for tex in _PENDING:
        tex.scene = scene
        n += 1
    _PENDING.clear()
    return n


def pending_count():
    """How many AO textures are waiting for a scene. Non-zero after a render means the
    binding step was skipped, which is otherwise silent until the first eval."""
    return len(_PENDING)


class AmbientOcclusion(TextureBase):
    '''Blender's Ambient Occlusion shader node.'''

    def __init__(self, props):
        TextureBase.__init__(self, props)
        self.channel = str(props.get('channel', 'Color'))
        if self.channel not in CHANNELS:
            raise RuntimeError("ambient_occlusion: output '%s' is not supported; %s are."
                               % (self.channel, ', '.join(CHANNELS)))
        self.samples = int(props.get('samples', 16))
        if self.samples < 1:
            raise RuntimeError('ambient_occlusion: samples must be >= 1')
        self.inside = bool(props.get('inside', False))
        if bool(props.get('only_local', False)):
            # Cycles restricts the trace to the shading point's own object; `Scene::ray_test`
            # has no such restriction, and quietly tracing against the whole scene is a
            # different -- darker -- picture.
            raise RuntimeError("ambient_occlusion: 'only_local' has no Mitsuba counterpart.")
        self.distance = props.get_unbounded_texture('distance', 1.0)
        self.color = props.get_texture('color', 1.0)
        self.scene = None
        _PENDING.append(self)

    def _ao(self, si, active):
        if self.scene is None:
            raise RuntimeError(
                'ambient_occlusion was evaluated before a scene was bound to it. Call '
                'mitsuba.python.plugins.textures.ambient_occlusion.bind_scene(scene) right '
                'after loading the scene.')

        max_dist = self.distance.eval_1(si, active)
        n = si.sh_frame.n
        if self.inside:
            n = -n
        frame = mi.Frame3f(dr.normalize(n))

        # Cycles' early-out: a non-positive radius means no sampling and full visibility.
        live = active & (max_dist > 0.0)

        px = mi.Float(si.p.x)
        py = mi.Float(si.p.y)
        pz = mi.Float(si.p.z)
        unoccluded = mi.Float(0.0)
        for k in range(self.samples):
            kf = mi.Float(float(k) + 1.0)
            u1 = hash_float4_to_float(px, py, pz, kf)
            u2 = hash_float4_to_float(kf, pz, py, px)
            # Malley: a uniform disk sample lifted to the hemisphere is cosine-weighted,
            # which is what Cycles' sample_uniform_disk + sqrt(1 - d.d) does.
            r = dr.sqrt(u1)
            phi = 2.0 * dr.pi * u2
            d = mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi),
                            dr.sqrt(dr.maximum(1.0 - u1, 0.0)))
            ray = si.spawn_ray(frame.to_world(d))
            ray.maxt = max_dist
            hit = self.scene.ray_test(ray, live)
            unoccluded += dr.select(live & ~hit, 1.0, 0.0)

        frac = unoccluded * (1.0 / self.samples)
        return dr.select(live, frac, 1.0)

    def eval_3(self, si, active=True):
        ao = self._ao(si, active)
        if self.channel == 'AO':
            return mi.Color3f(ao)
        return self.color.eval_3(si, active) * ao

    def eval_color3(self, si, active=True):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_1(self, si, active=True):
        if self.channel == 'AO':
            return self._ao(si, active)
        return mi.luminance(self.eval_3(si, active))

    def mean(self):
        return mi.Float(1.0) if self.channel == 'AO' else self.color.mean()

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return (f'AmbientOcclusion[channel={self.channel}, samples={self.samples}, '
                f'inside={self.inside}]')


mi.register_field('ambient_occlusion', lambda props: AmbientOcclusion(props))
