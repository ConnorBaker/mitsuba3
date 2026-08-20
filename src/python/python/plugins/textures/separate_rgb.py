from __future__ import annotations # Delayed parsing of type annotations
from typing import Tuple

import drjit as dr
import mitsuba as mi

from .blender_color import rgb_to_hsl, rgb_to_hsv

class SeparateRGB(mi.Texture):
    '''
    Separate RGB Blender shader node texture.
    '''
    def __init__(self, props):
        mi.Texture.__init__(self, props)
        # HSR: was `props.get('input')`, which does NOT promote a constant to a Texture --
        # so a Separate Color fed by a constant colour produced a Color3f here and faulted on
        # `.eval_3` at render time. `get_texture` promotes.
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        self.texture = props.get_unbounded_texture('input', 0.0)
        self.channel = props.get('channel', 'r')
        if self.channel not in ['r', 'g', 'b']:
            raise ValueError(f"SeparateRGB: Invalid channel {self.channel}")
        # HSR: Blender's Separate Color node has THREE modes and this plugin implemented
        # one. The other two are not relabelled channels -- HSL's lightness is the midpoint
        # of the extremes where HSV's value is the maximum, and their saturations differ
        # entirely -- so exporting either as RGB would have been a different picture.
        self.mode = str(props.get('mode', 'RGB')).upper()
        if self.mode not in ('RGB', 'HSV', 'HSL'):
            raise ValueError(f"SeparateRGB: Invalid mode {self.mode}")

    def traverse(self, cb):
        cb.put('input', self.texture, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        self.texture.parameters_changed(keys)

    def _eval_channel(self, si, active):
        color = self.texture.eval_3(si, active)
        if self.mode == 'HSV':
            color = rgb_to_hsv(color)
        elif self.mode == 'HSL':
            color = rgb_to_hsl(color)
        return color[{ 'r': 0, 'g': 1, 'b': 2 }[self.channel]]

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self._eval_channel(si, active))

    def eval_1(self, si, active):
        return mi.Float(self._eval_channel(si, active))

    def eval_3(self, si, active):
        return mi.Color3f(self._eval_channel(si, active))

    def mean(self):
        return self.texture.mean()

    def resolution(self):
        return self.texture.resolution()

    def is_spatially_varying(self):
        return self.texture.is_spatially_varying()

    def to_string(self):
        return f'Separate RGB[input={self.texture}, mode={self.mode}, channel={self.channel}]'

mi.register_texture('separate_rgb', lambda props: SeparateRGB(props))
