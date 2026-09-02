from __future__ import annotations # Delayed parsing of type annotations
from typing import Tuple

import drjit as dr
import mitsuba as mi
from ._base import TextureBase

class UVWrapper(TextureBase):
    '''
    Wrapper texture to evaluate input texture with a different set of UVs.

    This plugin in used in the Blender-Mitsuba add-on.
    '''
    def __init__(self, props):
        TextureBase.__init__(self, props)
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        # HSR: both used `props.get`, which does not promote a constant to a Texture,
        # so a constant on either socket faulted on `.eval_3` mid-render.
        self.uv = props.get_unbounded_texture('uv', 0.0)
        self.input = props.get_texture('input', 0.0)

    def eval_si(self, si, active):
        si = mi.SurfaceInteraction3f(si)
        si.uv = self.uv.eval_3(si, active).xy
        si.uv = mi.Vector2f(si.uv.x, 1.0-si.uv.y) # Blender convention
        return si

    def eval_color3(self, si, active):
        return self.input.eval(self.eval_si(si, active), active)

    def eval_1(self, si, active):
        return self.input.eval_1(self.eval_si(si, active), active)

    def eval_3(self, si, active):
        return self.input.eval_3(self.eval_si(si, active), active)

    def mean(self):
        return self.input.mean()

    def to_string(self):
        return f'UVWrapper[uv={self.uv}, input={self.input}]'

mi.register_field('uv_wrapper', lambda props: UVWrapper(props))
