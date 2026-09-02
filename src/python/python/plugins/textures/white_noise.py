# Blender's White Noise Texture node -- `kernel/svm/white_noise.h`.
#
# Not "noise" in the Perlin sense at all: there is no lattice and no interpolation, just
# Cycles' float hash applied to the coordinate, so neighbouring samples are uncorrelated.
# That makes it the one texture where getting the hash bit-exact is the WHOLE of getting the
# node right -- any other hash produces something equally white and entirely different.

import drjit as dr
import mitsuba as mi
from ._base import TextureBase

from .blender_hash import (hash_float_to_float, hash_float_to_float3, hash_float2_to_float,
                           hash_float2_to_float3, hash_float3_to_float,
                           hash_float3_to_float3, hash_float4_to_float,
                           hash_float4_to_float3)


class WhiteNoise(TextureBase):
    """White Noise Texture."""

    def __init__(self, props):
        TextureBase.__init__(self, props)
        self.dimensions = int(props.get('dimensions', 3))
        if self.dimensions not in (1, 2, 3, 4):
            raise Exception("white_noise: `dimensions` must be 1, 2, 3 or 4, got %d"
                            % self.dimensions)
        self.channel = str(props.get('channel', 'fac')).lower()
        if self.channel not in ('fac', 'color'):
            raise Exception("white_noise: `channel` must be 'fac' or 'color', got '%s'"
                            % self.channel)
        self.vector = None
        if 'vector' in [str(k) for k in props.keys()]:
            self.vector = props.get_unbounded_texture('vector', 0.0)
        self.w = props.get_unbounded_texture('w', 0.0)

    def _p(self, si, active):
        if self.vector is not None:
            v = self.vector.eval_3(si, active)
            return mi.Float(v.x), mi.Float(v.y), mi.Float(v.z)
        return mi.Float(si.uv.x), mi.Float(si.uv.y), mi.Float(0.0)

    def _value(self, si, active):
        x, y, z = self._p(si, active)
        w = self.w.eval_1(si, active)
        if self.dimensions == 1:
            return hash_float_to_float(w)
        if self.dimensions == 2:
            return hash_float2_to_float(x, y)
        if self.dimensions == 3:
            return hash_float3_to_float(x, y, z)
        return hash_float4_to_float(x, y, z, w)

    def _color(self, si, active):
        x, y, z = self._p(si, active)
        w = self.w.eval_1(si, active)
        if self.dimensions == 1:
            return hash_float_to_float3(w)
        if self.dimensions == 2:
            return hash_float2_to_float3(x, y)
        if self.dimensions == 3:
            return hash_float3_to_float3(x, y, z)
        return hash_float4_to_float3(x, y, z, w)

    def eval_color3(self, si, active=True):
        return self.eval_3(si, active)

    def eval_1(self, si, active=True):
        if self.channel == 'color':
            c = self._color(si, active)
            return (c.x + c.y + c.z) / 3.0
        return self._value(si, active)

    def eval_3(self, si, active=True):
        if self.channel == 'color':
            return self._color(si, active)
        return mi.Color3f(self._value(si, active))

    def mean(self):
        return 0.5

    def traverse(self, cb):
        cb.put('w', self.w, mi.ParamFlags.Differentiable)
        if self.vector is not None:
            cb.put('vector', self.vector, mi.ParamFlags.Differentiable)

    def to_string(self):
        return 'WhiteNoise[dimensions=%d, channel=%s]' % (self.dimensions, self.channel)


mi.register_field('white_noise', lambda props: WhiteNoise(props))
