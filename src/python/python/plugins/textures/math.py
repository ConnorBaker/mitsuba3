from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

# HSR: every one of these used to be spelled the obvious way and disagreed with Cycles on
# inputs a texture reaches constantly. They now come from `blender_math`, which transcribes
# Cycles' own `safe_*` / `compatible_*` helpers, so the scalar Math node and the Vector Math
# node cannot drift apart.
from .blender_math import (
    blender_round, blender_trunc, compare, compatible_atan2, compatible_sign,
    fract, inverse_sqrt, pingpong, safe_acos, safe_asin, safe_divide, safe_floored_modulo,
    safe_log, safe_modulo, safe_pow, safe_sqrt, smooth_min, wrap)

# Supported math operators
# See blender implementation: https://github.com/blender/blender/blob/594f47ecd2d5367ca936cf6fc6ec8168c2b360d0/source/blender/gpu/shaders/material/gpu_shader_material_math.glsl#L203
OPERATORS = {
    'ADD':            (lambda a, b, c: a + b),
    'SUBTRACT':       (lambda a, b, c: a - b),
    'MULTIPLY':       (lambda a, b, c: a * b),
    'DIVIDE':         (lambda a, b, c: safe_divide(a, b)),  # HSR: Blender returns 0 on b == 0
    'MULTIPLY_ADD':   (lambda a, b, c: a * b + c),
    'POWER':          (lambda a, b, c: safe_pow(a, b)),
    'LOGARITHM':      (lambda a, b, c: safe_log(a, b)),
    'SQRT':           (lambda a, b, c: safe_sqrt(a)),
    'INVERSE_SQRT':   (lambda a, b, c: inverse_sqrt(a)),
    'ABSOLUTE':       (lambda a, b, c: dr.abs(a)),
    'EXPONENT':       (lambda a, b, c: dr.exp(a)),
    'MINIMUM':        (lambda a, b, c: dr.minimum(a, b)),
    'MAXIMUM':        (lambda a, b, c: dr.maximum(a, b)),
    'LESS_THAN':      (lambda a, b, c: dr.select(a < b, 1.0, 0.0)),
    'GREATER_THAN':   (lambda a, b, c: dr.select(a > b, 1.0, 0.0)),
    # HSR: was `a >= 0 ? 1 : 0` -- so every NEGATIVE input returned 0 where Blender
    # returns -1. Rendered: Cycles -1, Mitsuba 0, over half the plane.
    'SIGN':           (lambda a, b, c: compatible_sign(a)),
    # HSR: the floor is FLT_EPSILON, not 1e-5 -- two orders of magnitude apart.
    'COMPARE':        (lambda a, b, c: compare(a, b, c)),
    'SMOOTH_MIN':     (lambda a, b, c: smooth_min(a, b, c)),
    'SMOOTH_MAX':     (lambda a, b, c: -smooth_min(-a, -b, c)),
    # HSR: `dr.round` is round-half-to-EVEN; Blender's is `floor(a + 0.5)`.
    'ROUND':          (lambda a, b, c: blender_round(a)),
    'FLOOR':          (lambda a, b, c: dr.floor(a)),
    'CEIL':           (lambda a, b, c: dr.ceil(a)),
    'TRUNC':          (lambda a, b, c: blender_trunc(a)),
    # HSR: Blender's enum identifier is 'FRACT'. 'FRACTION' was never a name Blender
    # emits, so a Fract node raised KeyError mid-render; kept as an alias.
    'FRACT':          (lambda a, b, c: fract(a)),
    'FRACTION':       (lambda a, b, c: fract(a)),
    'MODULO':         (lambda a, b, c: safe_modulo(a, b)),
    'FLOORED_MODULO': (lambda a, b, c: safe_floored_modulo(a, b)),
    'WRAP':           (lambda a, b, c: wrap(a, b, c)),
    'SNAP':           (lambda a, b, c: dr.floor(safe_divide(a, b)) * b),
    'PINGPONG':       (lambda a, b, c: pingpong(a, b)),
    'SINE':           (lambda a, b, c: dr.sin(a)),
    'COSINE':         (lambda a, b, c: dr.cos(a)),
    'TANGENT':        (lambda a, b, c: dr.tan(a)),
    # HSR: was `dr.select(a <= 1.0 & a >= -1.0, ...)`, which Python parses as
    # `a <= (1.0 & a) >= -1.0` -- a bitwise AND against a float. Blender CLAMPS.
    'ARCSINE':        (lambda a, b, c: safe_asin(a)),
    'ARCCOSINE':      (lambda a, b, c: safe_acos(a)),
    'ARCTANGENT':     (lambda a, b, c: dr.atan(a)),
    'ARCTAN2':        (lambda a, b, c: compatible_atan2(a, b)),
    'SINH':           (lambda a, b, c: dr.sinh(a)),
    'COSH':           (lambda a, b, c: dr.cosh(a)),
    'TANH':           (lambda a, b, c: dr.tanh(a)),
    'RADIANS':        (lambda a, b, c: dr.deg2rad(a)),
    'DEGREES':        (lambda a, b, c: dr.rad2deg(a))
}

class Math(mi.Texture):
    '''
    Math Blender shader node texture.
    '''
    def __init__(self, props):
        mi.Texture.__init__(self, props)
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        self.input_0 = props.get_unbounded_texture('input_0')
        self.input_1 = props.get_unbounded_texture('input_1', 0.0)
        self.input_2 = props.get_unbounded_texture('input_2', 0.0)
        self.mode = str(props.get('mode'))
        self.clamp = props.get('clamp', False)

        if not self.mode in OPERATORS.keys():
            mi.Log(mi.LogLevel.Error, f'Unknown math operator: {self.mode}')

    def traverse(self, cb):
        cb.put('input_0', self.input_0, +mi.ParamFlags.Differentiable)
        cb.put('input_1', self.input_1, +mi.ParamFlags.Differentiable)
        cb.put('input_2', self.input_2, +mi.ParamFlags.Differentiable)  # HSR: was input_1

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self.process(
            self.input_0.eval(si, active),
            self.input_1.eval(si, active),
            self.input_2.eval(si, active)
        ))

    def eval_1(self, si, active):
        return mi.Float(self.process(
            self.input_0.eval_1(si, active),
            self.input_1.eval_1(si, active),
            self.input_2.eval_1(si, active)
        ))

    def eval_3(self, si, active):
        return mi.Color3f(self.process(
            self.input_0.eval_3(si, active),
            self.input_1.eval_3(si, active),
            self.input_2.eval_3(si, active)
        ))

    def process(self, a, b, c):
        out = OPERATORS[self.mode](a, b, c)

        if self.clamp:
            out = dr.clip(out, 0.0, 1.0)

        return out

    def mean(self):
        return self.input_0.mean() # TODO best effort

    def resolution(self):
        return self.input_0.resolution()

    def is_spatially_varying(self):
        return any([t.is_spatially_varying() for t in [self.input_0, self.input_1, self.input_2]])

    def to_string(self):
        return f'Math[input_0={self.input_0}, input_1={self.input_1}, input_2={self.input_2}, mode={self.mode}, clamp={self.clamp}]'

mi.register_texture('math', lambda props: Math(props))
