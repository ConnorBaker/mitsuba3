from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
from drjit.auto import Float, Array3f, Int
import mitsuba as mi
import numpy as np

'''
HSR: CYCLES BAKES CURVE MAPPINGS; IT DOES NOT EVALUATE BEZIER HANDLES IN THE KERNEL.

`intern/cycles/blender/shader.cpp` builds an `RGBCurvesNode` from
`curvemapping_color_to_array(mapping, curves, RAMP_TABLE_SIZE, true)` plus
`curvemapping_minmax(mapping, 4, &min_x, &max_x)` and
`set_extrapolate(mapping.flag & CUMA_EXTEND_EXTRAPOLATE)`. `curvemapping_color_to_array`
(`intern/cycles/blender/util.h`) evaluates, for i in 0..RAMP_TABLE_SIZE,

    t       = min_x + i / size * (max_x - min_x)
    entry_i = ( R(I(t)), G(I(t)), B(I(t)) )                     [rgb_curve, 4 curves]
    entry_i = ( R(t),    G(t),    B(t)    )                     [vector curve, 3 curves]

with each of R/G/B/I being `BKE_curvemap_evaluateF` -- which is exactly what Blender's
`CurveMapping.evaluate(curve, t)` calls. The kernel side (`kernel/svm/ramp.h`,
`svm_node_curves`) then only does

    relpos = (color - min_x) / (max_x - min_x)
    out    = lerp(color, per-channel rgb_ramp_lookup(relpos, interpolate=true, extrapolate),
                  fac)

So the exporter bakes the table and this plugin is the lookup. That is exact for bezier
handles, for `AUTO` / `AUTO_CLAMPED` / `VECTOR` handle types, and for the curve
extrapolation mode -- none of which are reimplemented here. The previous implementation
interpolated the CONTROL POINTS piecewise-linearly, which is right only for a curve whose
every segment is already a straight line.

One deliberate divergence from Blender's UI, which is parity with CYCLES rather than a
fallback: `CurveMapping.tone` (`STANDARD` / `FILMLIKE`) is NOT applied. Tone lives in
`BKE_curvemapping_evaluate_premulRGBF` (`source/blender/blenkernel/intern/colortools.cc`);
`BKE_curvemap_evaluateF`, which is what Cycles calls and what the bake above calls, does
not consult it. A FILMLIKE RGB Curves node therefore renders in Cycles as if it were
STANDARD, and this reproduces that.
'''


def _lookup3(table, n, f, extrapolate):
    '''`rgb_ramp_lookup(ramp, f, interpolate=true, extrapolate, table_size)` verbatim
    (intern/cycles/kernel/svm/ramp_util.h), branch-free.'''
    fc = dr.clip(f, 0.0, 1.0) * (n - 1)
    i = dr.clip(Int(fc), 0, n - 1)
    t = fc - Float(i)
    e0 = dr.gather(Array3f, table, i)
    e1 = dr.gather(Array3f, table, dr.minimum(i + 1, n - 1))
    w = dr.maximum(t, 0.0)
    res = (1.0 - w) * e0 + w * e1

    if not extrapolate:
        return res

    # The extrapolating arms: below 0 it reflects about entry 0, above 1 about entry n-1,
    # with the slope taken from the neighbouring entry and re-scaled by (n - 1).
    lo0 = dr.gather(Array3f, table, Int(0))
    lo1 = dr.gather(Array3f, table, Int(1))
    hi0 = dr.gather(Array3f, table, Int(n - 1))
    hi1 = dr.gather(Array3f, table, Int(n - 2))
    below = (lo0 - lo1) * (-f) * (n - 1) + lo0
    above = (hi0 - hi1) * (f - 1.0) * (n - 1) + hi0
    res = dr.select(f < 0.0, below, res)
    res = dr.select(f > 1.0, above, res)
    return res


def _lookup1(table, n, f, extrapolate):
    '''`float_ramp_lookup`, the scalar twin of the above.'''
    fc = dr.clip(f, 0.0, 1.0) * (n - 1)
    i = dr.clip(Int(fc), 0, n - 1)
    t = fc - Float(i)
    e0 = dr.gather(Float, table, i)
    e1 = dr.gather(Float, table, dr.minimum(i + 1, n - 1))
    w = dr.maximum(t, 0.0)
    res = (1.0 - w) * e0 + w * e1

    if not extrapolate:
        return res

    lo0 = dr.gather(Float, table, Int(0))
    lo1 = dr.gather(Float, table, Int(1))
    hi0 = dr.gather(Float, table, Int(n - 1))
    hi1 = dr.gather(Float, table, Int(n - 2))
    below = (lo0 - lo1) * (-f) * (n - 1) + lo0
    above = (hi0 - hi1) * (f - 1.0) * (n - 1) + hi0
    res = dr.select(f < 0.0, below, res)
    res = dr.select(f > 1.0, above, res)
    return res


def _parse_table(s, stride, who):
    flat = np.asarray([float(x) for x in s.split()], dtype=np.float32)
    if flat.size == 0 or flat.size % stride != 0:
        raise ValueError(f'{who}: `table` holds {flat.size} floats, which is not a non-zero '
                         f'multiple of {stride}.')
    n = flat.size // stride
    if n < 2:
        raise ValueError(f'{who}: `table` needs at least 2 entries to be looked up.')
    return n, flat


class FloatCurve(mi.Texture):
    '''
    Float curve Blender shader node texture -- Cycles' `FloatCurveNode` / `svm_node_curve`.
    '''
    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.factor = props.get_texture('factor', 1.0)
        self.value  = props.get_texture('value')
        table = props.get('table', '')
        if not table:
            raise RuntimeError('FloatCurve: `table` (the baked curve, one float per entry) '
                               'is required. The control-point input this replaced could '
                               'not express bezier handles.')
        self.n, flat = _parse_table(table, 1, 'FloatCurve')
        self.table = Float(flat)
        self.min_x = float(props.get('min_x', 0.0))
        self.max_x = float(props.get('max_x', 1.0))
        self.extrapolate = bool(props.get('extrapolate', True))
        if self.max_x == self.min_x:
            raise ValueError('FloatCurve: `max_x` equals `min_x`; the curve has no domain.')

    def traverse(self, cb):
        cb.put('factor', self.factor, +mi.ParamFlags.Differentiable)
        cb.put('value',  self.value,  +mi.ParamFlags.Differentiable)

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self.process(si, active))

    def eval_1(self, si, active):
        return mi.Float(self.process(si, active))

    def eval_3(self, si, active):
        return mi.Color3f(self.process(si, active))

    def process(self, si, active):
        factor = self.factor.eval_1(si, active)
        value  = self.value.eval_1(si, active)
        relpos = (value - self.min_x) / (self.max_x - self.min_x)
        v = _lookup1(self.table, self.n, relpos, self.extrapolate)
        return dr.lerp(value, v, factor)

    def mean(self):
        return self.value.mean() # TODO best effort

    def resolution(self):
        return self.value.resolution()

    def is_spatially_varying(self):
        return any([t.is_spatially_varying() for t in [self.factor, self.value]])

    def to_string(self):
        return f'FloatCurve[factor={self.factor}, value={self.value}, table_size={self.n}]'

mi.register_texture('float_curve', lambda props: FloatCurve(props))


class RGBCurve(mi.Texture):
    '''
    RGB curve Blender shader node texture -- Cycles' `RGBCurvesNode` / `svm_node_curves`.
    '''
    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.factor = props.get_texture('factor', 1.0)
        self.color  = props.get_texture('color')
        table = props.get('table', '')
        if not table:
            raise RuntimeError('RGBCurve: `table` (the baked curve, three floats per entry) '
                               'is required. The control-point input this replaced could '
                               'not express bezier handles, nor the combined C-then-RGB '
                               'evaluation order Cycles bakes in.')
        self.n, flat = _parse_table(table, 3, 'RGBCurve')
        # 3 x N, so `dr.gather(Array3f, ...)` picks out one entry.
        self.table = Array3f(flat.reshape(self.n, 3).T)
        self.min_x = float(props.get('min_x', 0.0))
        self.max_x = float(props.get('max_x', 1.0))
        self.extrapolate = bool(props.get('extrapolate', True))
        if self.max_x == self.min_x:
            raise ValueError('RGBCurve: `max_x` equals `min_x`; the curve has no domain.')

    def traverse(self, cb):
        cb.put('factor', self.factor, +mi.ParamFlags.Differentiable)
        cb.put('color',  self.color,  +mi.ParamFlags.Differentiable)

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self.process(self.color.eval(si, active), si, active))

    def eval_1(self, si, active):
        return mi.luminance(self.process(mi.Color3f(self.color.eval_1(si, active)), si, active))

    def eval_3(self, si, active):
        return mi.Color3f(self.process(mi.Color3f(self.color.eval_3(si, active)), si, active))

    def process(self, color, si, active):
        factor = self.factor.eval_1(si, active)
        inv = 1.0 / (self.max_x - self.min_x)
        # One lookup per channel, each keeping only its OWN component -- the table's three
        # rows are R(I(t)), G(I(t)), B(I(t)) evaluated at the SAME t, so channel c must be
        # read at channel c's own relative position.
        r = _lookup3(self.table, self.n, (color.x - self.min_x) * inv, self.extrapolate)[0]
        g = _lookup3(self.table, self.n, (color.y - self.min_x) * inv, self.extrapolate)[1]
        b = _lookup3(self.table, self.n, (color.z - self.min_x) * inv, self.extrapolate)[2]
        return dr.lerp(color, mi.Color3f(r, g, b), factor)

    def mean(self):
        return self.color.mean() # TODO best effort

    def resolution(self):
        return self.color.resolution()

    def is_spatially_varying(self):
        return any([t.is_spatially_varying() for t in [self.factor, self.color]])

    def to_string(self):
        return f'RGBCurve[factor={self.factor}, color={self.color}, table_size={self.n}]'

mi.register_texture('rgb_curve', lambda props: RGBCurve(props))
