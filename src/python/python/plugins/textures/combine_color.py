from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
from ._base import TextureBase

from .blender_color import hsl_to_rgb, hsv_to_rgb

class CombineColor(TextureBase):
    '''
    RGB to BW Blender shader node texture.
    '''
    def __init__(self, props):
        TextureBase.__init__(self, props)
        self.mode = str(props.get('mode', 'RGB')).upper()
        if self.mode not in ('RGB', 'HSV', 'HSL'):
            # HSR: was `assert self.mode == 'RGB'`, i.e. an AssertionError with no message
            # in the middle of a scene load. HSV and HSL are now implemented rather than
            # refused.
            raise ValueError(f"CombineColor: Invalid mode {self.mode}")
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        self.R = props.get_unbounded_texture('red', 0.0)
        self.G = props.get_unbounded_texture('green', 0.0)
        self.B = props.get_unbounded_texture('blue', 0.0)

    def traverse(self, cb):
        cb.put('R', self.R, +mi.ParamFlags.Differentiable)
        cb.put('G', self.G, +mi.ParamFlags.Differentiable)
        cb.put('B', self.B, +mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def eval_color3(self, si, active):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_3(self, si, active):
        c = mi.Color3f(
            self.R.eval_1(si, active),
            self.G.eval_1(si, active),
            self.B.eval_1(si, active)
        )
        if self.mode == 'HSV':
            return hsv_to_rgb(c)
        if self.mode == 'HSL':
            return hsl_to_rgb(c)
        return c

    def eval_1(self, si, active):
        c = self.eval_3(si, active)
        return (c.x + c.y + c.z) / 3.0

    def mean(self):
        return self.color.mean() # TODO best effort

    def resolution(self):
        return self.color.resolution()

    def is_spatially_varying(self):
        return any([t.is_spatially_varying() for t in [self.R, self.G, self.B]])

    def to_string(self):
        return f'CombineColor[mode={self.mode}, R={self.R}, G={self.G}, B={self.B}]'

mi.register_field('combine_color', lambda props: CombineColor(props))
