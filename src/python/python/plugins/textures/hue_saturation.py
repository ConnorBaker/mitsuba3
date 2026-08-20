from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi


def wrap(a, b):
    """Floored modulo: the result carries `b`'s sign, so a NEGATIVE input WRAPS.

    The helper this replaces was `(a - Int32(a)) + Int32(a) % b`, and its caller reached for
    `dr.abs` to keep it in range. `abs` REFLECTS a negative hue instead of wrapping it, which
    maps -t to +t and therefore swaps green and blue for every hue the Hue input pushes below
    zero. Measured, not deduced: with hue 0.32 on a red-green sweep, Cycles drew (0.76, 0,
    0.79) where this plugin drew (0.76, 0.79, 0).
    """
    return a - b * dr.floor(a / b)


def rgb2hsv(rgb):
    """Blender's `rgb_to_hsv`, hue in [0, 1].

    Rewritten from the six-sector cascade of masked assignments this replaces. That form
    relied on the sector predicates being mutually exclusive at the boundaries, where they
    are not (`R >= G >= B` and `R >= B >= G` are both true when G == B), so a later branch
    could silently overwrite an earlier one. This states the sextant once.
    """
    r, g, b = rgb.x, rgb.y, rgb.z
    mx = dr.maximum(r, dr.maximum(g, b))
    mn = dr.minimum(r, dr.minimum(g, b))
    delta = mx - mn
    safe_delta = dr.select(delta > 0.0, delta, 1.0)
    h_r = wrap((g - b) / safe_delta, 6.0)
    h_g = (b - r) / safe_delta + 2.0
    h_b = (r - g) / safe_delta + 4.0
    h = dr.select(delta <= 0.0, 0.0,
                  dr.select(mx == r, h_r, dr.select(mx == g, h_g, h_b))) / 6.0
    s = dr.select(mx > 0.0, delta / dr.select(mx > 0.0, mx, 1.0), 0.0)
    return mi.Color3f(h, s, mx)


def hsv2rgb(hsv):
    """Blender's `hsv_to_rgb`: three clamped triangle waves of the hue, hue in [0, 1]."""
    h = wrap(hsv.x, 1.0)
    s, v = hsv.y, hsv.z
    nr = dr.clip(dr.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    ng = dr.clip(2.0 - dr.abs(h * 6.0 - 2.0), 0.0, 1.0)
    nb = dr.clip(2.0 - dr.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return mi.Color3f(((nr - 1.0) * s + 1.0) * v,
                      ((ng - 1.0) * s + 1.0) * v,
                      ((nb - 1.0) * s + 1.0) * v)


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
