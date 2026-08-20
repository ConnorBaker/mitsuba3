from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi


# HSR: these three used to be DEFINED here and imported by `mix_color` across a plugin
# boundary. They are Blender's, they are shared by four nodes, and they now live in
# `blender_color` with the HSL pair beside them.
from .blender_color import hsv_to_rgb as hsv2rgb
from .blender_color import rgb_to_hsv as rgb2hsv
from .blender_color import wrap01 as wrap


class HueSaturationValue(mi.Texture):
    '''
    Hue-Saturation-Value Blender shader node texture.
    '''
    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.hue        = props.get_texture('hue', 0.5)
        self.saturation = props.get_texture('saturation', 1.0)
        self.value      = props.get_texture('value', 1.0)
        self.mix        = props.get_texture('mix', 1.0)
        self.input      = props.get_texture('input', 1.0)

    def traverse(self, cb):
        cb.put('hue',        self.hue,        +mi.ParamFlags.Differentiable)
        cb.put('saturation', self.saturation, +mi.ParamFlags.Differentiable)
        cb.put('value',      self.value,      +mi.ParamFlags.Differentiable)
        cb.put('mix',        self.mix,        +mi.ParamFlags.Differentiable)
        cb.put('input',      self.input,      +mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_1(self, si, active):
        return mi.Float(self.input.eval_1(si, active) * self.value.eval_1(si, active))

    def eval_3(self, si, active):
        hue        = self.hue.eval_1(si, active)
        saturation = self.saturation.eval_1(si, active)
        value      = self.value.eval_1(si, active)
        mix        = self.mix.eval_1(si, active)
        color      = self.input.eval_3(si, active)

        hsv = rgb2hsv(color)

        # Cycles: `h = fract(h + hue + 0.5)`, then a clamp of the result at zero before the
        # mix. The hue is in [0, 1] now, not degrees.
        hsv.x = wrap(hsv.x + hue + 0.5, 1.0)
        hsv.y = dr.clip(hsv.y * saturation, 0.0, 1.0)
        hsv.z *= value

        shifted = dr.maximum(hsv2rgb(hsv), 0.0)
        return mi.Color3f(dr.lerp(color, shifted, mix))

    def mean(self):
        return self.input.mean() # TODO best effort

    def resolution(self):
        return self.input.resolution()

    def is_spatially_varying(self):
        return any([t.is_spatially_varying() for t in [
            self.hue, self.saturation, self.value, self.mix, self.input
        ]])

    def to_string(self):
        return f'HueSaturationValue[hue={self.hue}, saturation={self.saturation}, value={self.value}, mix={self.mix}, color={self.color}]'

mi.register_texture('hue_saturation_value', lambda props: HueSaturationValue(props))
