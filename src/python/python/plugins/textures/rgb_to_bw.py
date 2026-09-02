from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
from ._base import TextureBase

class RGB2BW(TextureBase):
    '''
    RGB to BW Blender shader node texture.
    '''
    def __init__(self, props):
        TextureBase.__init__(self, props)
        self.color = props.get_texture('color', 0.5)

    def traverse(self, cb):
        cb.put('color', self.color, +mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def eval_color3(self, si, active):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_1(self, si, active):
        return mi.Float(mi.luminance(self.color.eval_3(si, active)))

    def eval_3(self, si, active):
        return mi.Color3f(mi.luminance(self.color.eval_3(si, active)))

    def mean(self):
        return mi.luminance(self.color.mean()) # TODO best effort

    def resolution(self):
        return self.color.resolution()

    def is_spatially_varying(self):
        return self.color.is_spatially_varying()

    def to_string(self):
        return f'RGB2BW[color={self.color}]'

mi.register_field('rgb_to_bw', lambda props: RGB2BW(props))
