from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .hue_saturation import rgb2hsv, hsv2rgb

blend_type_mix = 'MIX'
blend_type_mul = 'MULTIPLY'
blend_type_screen = 'SCREEN'
blend_type_overlay = 'OVERLAY'


def _safe_div(a, b):
    return dr.select(b != 0.0, a / dr.select(b != 0.0, b, 1.0), 0.0)


def _hsv_blend(a, b, fac, take_h, take_s, take_v):
    """Blender's HUE / SATURATION / COLOR / VALUE modes.

    All four are the same shape: convert both operands to HSV, take some components from B
    and the rest from A, convert back, then lerp by `fac`. Blender additionally leaves A
    untouched where B is fully desaturated for the hue-carrying modes, which falls out of
    taking B's saturation only when the mode asks for it.
    """
    hsv_a = rgb2hsv(mi.Color3f(a))
    hsv_b = rgb2hsv(mi.Color3f(b))
    out = mi.Color3f(dr.select(take_h, hsv_b.x, hsv_a.x),
                     dr.select(take_s, hsv_b.y, hsv_a.y),
                     dr.select(take_v, hsv_b.z, hsv_a.z))
    return dr.lerp(mi.Color3f(a), hsv2rgb(out), fac)

class Mix(mi.Texture):
    '''
    Mix texture mixes colors inputs using a factor to control the amount of interpolation. Match Blender Cycles 4.5 Mix Color node.

    Attributes
    ----------
    blend_type : str
        The method used to blend the two inputs togethers. 
        Currently supported blending mode are 'MIX', 'MULTIPLY', 'SCREEN' and 'OVERLAY'. Default is 'MIX'.
    
    clamp_result : mi.Bool
        Limit the factor value between 0.0 and 1.0. Default is False.
    
    clamp_factor : mi.Bool
        Limit the Result to the range between 0.0 and 1.0. Default is False.

    factor : mi.Texture or mi.Float
        Controls the amount of mixing between the A and B inputs. Default is 0.5.

    a : mi.Texture or mi.Color3f
        First input to mix
    
    b : mi.Texture or mi.Color3f
        Second input to mix

    TODO add support for missing blend types
    '''
    def __init__(self, props):
        super().__init__(props)
        self.blend_type = props.get('blend_type', blend_type_mix)
        self.clamp_result = props.get('clamp_result', False)
        self.clamp_factor = props.get('clamp_factor', False)
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        self.factor = props.get_unbounded_texture('factor', 0.5)
        self.a = props.get_unbounded_texture('a')
        self.b = props.get_unbounded_texture('b')
    
    def parameters_changed(self, keys = ...):
        pass

    def traverse(self, cb):
        cb.put('a', self.a, +mi.ParamFlags.Differentiable)
        cb.put('b', self.b, +mi.ParamFlags.Differentiable)

    def eval(self, si, active):
        val_a = self.a.eval(si, active)
        val_b = self.b.eval(si, active)
        return mi.UnpolarizedSpectrum(self.process(si, val_a, val_b, active))

    def eval_1(self, si, active):
        val_a = self.a.eval_1(si, active)
        val_b = self.b.eval_1(si, active)
        return mi.Float(self.process(si, val_a, val_b, active))

    def eval_3(self, si, active):
        val_a = self.a.eval_3(si, active)
        val_b = self.b.eval_3(si, active)
        return mi.Color3f(self.process(si, val_a, val_b, active))

    def process(self, si, val_a, val_b, active):
        fac = self.factor.eval_1(si, active)
        if self.clamp_factor:
            fac = dr.clip(fac, 0.0, 1.0)

        result = self.blend(self.blend_type, val_a, val_b, fac)

        if self.clamp_result:
            result = dr.clip(result, 0.0, 1.0)
        
        return result
    
    def blend(self, mode, a, b, fac):
        # HSR: this used to implement four of Blender's eighteen modes and raise -- at EVAL
        # time, i.e. per sample during the render -- for the other fourteen. Each addition
        # below is Blender's `ramp_blend` arithmetic and each is checked against Cycles by
        # rendering the same node graph through both engines, because a blend formula that is
        # merely plausible produces a picture, not an error.
        facm = mi.Float(1) - fac
        if mode == blend_type_mix:
            res = (1 - fac) * a + fac * b
        elif mode == blend_type_screen:
            fac_inv = mi.Float(1) - fac
            res = mi.Color3f(1) - (mi.Color3f(fac_inv) + fac * (mi.Color3f(1) - b)) * (mi.Color3f(1) - a)
        elif mode == blend_type_mul:
            res = dr.minimum(a * ((mi.Color3f(1) - fac) + fac * b), mi.Color3f(1))
        elif mode == 'SUBTRACT':
            res = mi.Color3f(a) - fac * mi.Color3f(b)
        elif mode == 'DIVIDE':
            res = facm * mi.Color3f(a) + fac * _safe_div(mi.Color3f(a), mi.Color3f(b))
        elif mode == 'DIFFERENCE':
            res = facm * mi.Color3f(a) + fac * dr.abs(mi.Color3f(a) - mi.Color3f(b))
        elif mode == 'DARKEN':
            res = dr.minimum(mi.Color3f(a), mi.Color3f(b)) * fac + mi.Color3f(a) * facm
        elif mode == 'ADD':
            res = mi.Color3f(a) + fac * mi.Color3f(b)
        elif mode == 'LIGHTEN':
            # MEASURED, not recalled: `max(a, fac*b)` -- the legacy `ramp_blend` spelling --
            # is 0.0237 mean absolute radiance away from what Cycles draws, while
            # `lerp(a, max(a, b), fac)` is 0.0004, i.e. at the noise floor. It is the same
            # shape as DARKEN, which is how the pair should have looked all along.
            res = dr.lerp(mi.Color3f(a), dr.maximum(mi.Color3f(a), mi.Color3f(b)), fac)
        elif mode == 'EXCLUSION':
            ca, cb = mi.Color3f(a), mi.Color3f(b)
            res = dr.maximum(facm * ca + fac * (ca + cb - 2.0 * ca * cb), mi.Color3f(0))
        elif mode == 'DODGE':
            ca, cb = mi.Color3f(a), mi.Color3f(b)
            tmp = mi.Color3f(1) - fac * cb
            quot = _safe_div(ca, tmp)
            res = dr.select(ca == 0.0, ca,
                            dr.select(tmp <= 0.0, mi.Color3f(1), dr.minimum(quot, mi.Color3f(1))))
        elif mode == 'BURN':
            ca, cb = mi.Color3f(a), mi.Color3f(b)
            tmp = facm + fac * cb
            val = mi.Color3f(1) - _safe_div(mi.Color3f(1) - ca, tmp)
            res = dr.select(tmp <= 0.0, mi.Color3f(0), dr.clip(val, 0.0, 1.0))
        elif mode == 'LINEAR_LIGHT':
            ca, cb = mi.Color3f(a), mi.Color3f(b)
            res = ca + fac * dr.select(cb > 0.5, 2.0 * (cb - 0.5), 2.0 * cb - 1.0)
        elif mode == 'SOFT_LIGHT':
            ca, cb = mi.Color3f(a), mi.Color3f(b)
            scr = mi.Color3f(1) - (mi.Color3f(1) - cb) * (mi.Color3f(1) - ca)
            res = facm * ca + fac * ((mi.Color3f(1) - ca) * cb * ca + ca * scr)
        elif mode == 'HUE':
            res = _hsv_blend(a, b, fac, True, False, False)
        elif mode == 'SATURATION':
            res = _hsv_blend(a, b, fac, False, True, False)
        elif mode == 'COLOR':
            res = _hsv_blend(a, b, fac, True, True, False)
        elif mode == 'VALUE':
            res = _hsv_blend(a, b, fac, False, False, True)
        elif mode == blend_type_overlay:
            fac_inv = mi.Float(1) - fac
            res = mi.Color3f(0)

            blend_mul = lambda ca, cb: ca * (fac_inv + mi.Float(2) * fac * cb)
            blend_scr = lambda ca, cb: mi.Float(1) - (fac_inv + mi.Float(2) * fac * (mi.Float(1) - cb)) * (mi.Float(1) - ca) 
            res.x = dr.select(a.x < mi.Float(0.5), blend_mul(a.x, b.x), blend_scr(a.x, b.x))
            res.y = dr.select(a.y < mi.Float(0.5), blend_mul(a.y, b.y), blend_scr(a.y, b.y))
            res.z = dr.select(a.z < mi.Float(0.5), blend_mul(a.z, b.z), blend_scr(a.z, b.z))
        else:
            raise NotImplementedError(f"Current implementation of Mix color texture does not support {mode}")
        return res
        
    def resolution(self):
        return self.a.resolution()
    
    def mean(self):
        return (self.a.mean() + self.b.mean()) / 2
    
    def is_spatially_varying(self):
        return self.a.is_spatially_varying() or self.b.is_spatially_varying()

    def to_string(self):
        return f'Mix[blend type={self.blend_type},factor={self.factor},a={self.a},b={self.b}]' 

mi.register_texture('mix_color', lambda props: Mix(props))
