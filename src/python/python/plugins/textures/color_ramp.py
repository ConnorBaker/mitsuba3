from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
from drjit.auto import Float, Array4f, Int
import mitsuba as mi
import numpy as np

color_mode_RGB = 'RGB'
color_mode_HSV = 'HSV'
color_mode_HSL = 'HSL'

inter_ease = 'EASE'
inter_card = 'CARDINAL'
inter_bspline = 'B_SPLINE'
inter_const = 'CONSTANT'
inter_lin = 'LINEAR'

hue_inter_near = 'NEAR'
hue_inter_far = 'FAR'
hue_inter_cw = 'CW'
hue_inter_cww = 'CWW'

class Ramp(mi.Texture):
    '''
    Color ramp texture mapping values to colors using a gradient. Match Blender Cycles 4.5 Color Ramp node.

    Attributes
    ----------
    table : str
        The BAKED ramp: whitespace-separated floats, four per entry (r g b a), for
        `table_size` entries. This is the preferred input and it is what Cycles itself
        uses -- see the class notes below.

    interpolate : bool
        Whether the baked table is interpolated between entries. Cycles sets this to
        `ipotype != CONSTANT`. Only meaningful together with `table`. Default is True.

    color_mode : str
        Select the color mode to be used. Only 'RGB' is currently supported by the
        `elements` path. Default is 'RGB'. Ignored when `table` is given.

    interpolation : str
        Select the interpolation method to be used when color mode is 'RGB'.
        If hue_interpolation is specified, it is not used. Default is 'LINEAR'.
        Ignored when `table` is given.

    hue_interpolation : str
        Select the interpolation method to be used when color mode is 'HSV' or 'HSL'.
        If interpolation is specified, it is not used. Default is 'NEAR'.
        Ignored when `table` is given.

    fac : mi.Texture or mi.Float
        The value to map. Default is 0.5.

    elements : str
        String with the following format "e1-e2-e3-e4-..." where eN = "eN[1] eN[2] ... eN[4]"
        Represent the elements composing the color ramp.

    HSR: CYCLES DOES NOT EVALUATE A COLORBAND IN THE KERNEL, AND NEITHER SHOULD THIS.
    `intern/cycles/blender/shader.cpp` calls `colorramp_to_array` (`blender/util.h`), which
    walks `BKE_colorband_evaluate(&ramp, i / size, ...)` for i in 0..RAMP_TABLE_SIZE and
    ships the resulting `RAMP_TABLE_SIZE + 1` RGBA entries to the device, together with
    `set_interpolate(ipotype != COLBAND_INTERP_CONSTANT)`. The kernel side
    (`kernel/svm/ramp.h`, `rgb_ramp_lookup` with `extrapolate = false`) is then nothing but
    a clamped lookup with one lerp.

    So the `table` path below is BIT-FOR-BIT the Cycles evaluation, and it is exact for
    every interpolation mode (EASE / CARDINAL / B_SPLINE / CONSTANT / LINEAR) and every
    color mode (RGB / HSV / HSL, all four hue directions) WITHOUT reimplementing any of
    them -- Blender does the interpolation when the exporter bakes the table. The
    `elements` path is kept only for callers that still pass stops; it implements
    RGB + LINEAR and raises on anything else, which is why it must not be the default.
    '''
    def __init__(self, props):
        super().__init__(props)
        self.mode = props.get('color_mode', color_mode_RGB)
        self.inter = props.get('interpolation', inter_lin)
        self.hue_inter = props.get('hue_interpolation', hue_inter_near)
        self.fac = props.get_texture('fac', 0.5)

        self.table = None
        self.elems = None
        self.interpolate = bool(props.get('interpolate', True))
        table = props.get('table', '')
        elements = props.get('elements', '')
        if table:
            self.parse_table(table)
        elif elements:
            # The stop-list path implements exactly one combination. Refuse the rest HERE,
            # at scene-construction time, where the message is attached to the plugin that
            # cannot be built -- not per-sample from inside a traced kernel.
            if self.mode != color_mode_RGB:
                raise NotImplementedError(
                    f'ColorRamp: the `elements` path does not support color mode '
                    f'{self.mode}. Pass a baked `table` instead (that path supports every '
                    f'color mode, because Blender bakes it).')
            if self.inter != inter_lin:
                raise NotImplementedError(
                    f'ColorRamp: the `elements` path does not support interpolation mode '
                    f'{self.inter}. Pass a baked `table` instead (that path supports every '
                    f'interpolation mode, because Blender bakes it).')
            self.parse_elements(elements)
        else:
            raise RuntimeError('ColorRamp: neither `table` (preferred, the baked Cycles '
                               'ramp) nor `elements` (stops) was given.')

    def parse_table(self, s):
        flat = np.asarray([float(x) for x in s.split()], dtype=np.float32)
        if flat.size == 0 or flat.size % 4 != 0:
            raise ValueError(f'ColorRamp: `table` holds {flat.size} floats, which is not a '
                             'non-zero multiple of 4 (r g b a per entry).')
        self.table_size = flat.size // 4
        if self.table_size < 2:
            raise ValueError('ColorRamp: `table` needs at least 2 entries to be looked up.')
        # 4 x N, so a `dr.gather(Array4f, ...)` picks out one entry -- the same layout the
        # `elements` path uses.
        self.table = Array4f(flat.reshape(self.table_size, 4).T)

    def parse_elements(self, str):
        temp_elem = []
        temp_pos = []
        for s in str.split('-'):
            elem = []
            for e in s.split(' '):
                elem.append(float(e))
            temp_elem.append(elem[0:4])
            temp_pos.append(elem[4])
        self.elems = Array4f(np.asarray(temp_elem).T)
        self.positions = Float(temp_pos)

    def get_element_color(self, idx):
        return mi.Color3f(dr.gather(Array4f, self.elems, idx)[1:4])

    def get_element_alpha(self, idx):
        return mi.Float(dr.gather(Array4f, self.elems, idx)[0])

    def get_element_pos(self, idx):
        return mi.Float(dr.gather(Float, self.positions, idx))

    def get_element_len(self):
        return self.positions.shape[0]

    def parameters_changed(self, keys = ...):
        pass

    def traverse(self, cb):
        cb.put('fac', self.fac, +mi.ParamFlags.Differentiable)

    def eval_1(self, si, active):
        _, res = self.process(si, active)
        return res

    def eval_3(self, si, active):
        res, _ = self.process(si, active)
        return res

    def eval(self, si, active = True):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def process(self, si, active):
        # The dispatch is a PYTHON-level choice made once at construction, so it is kept
        # out of the traced region. It used to sit inside `@dr.syntax`, which turned two
        # `raise NotImplementedError` arms into `dr.if_stmt` branches -- so an unsupported
        # ramp faulted mid-render, from inside a kernel, instead of at scene-load time.
        if self.table is not None:
            return self.process_table(si, active)
        return self.process_elements(si, active)

    def process_table(self, si, active):
        # `rgb_ramp_lookup(ramp, f, interpolate, extrapolate=false, table_size)` verbatim
        # (intern/cycles/kernel/svm/ramp.h). Cycles' RGB ramp never extrapolates.
        fac = self.fac.eval_1(si, active)
        n = self.table_size
        f = dr.clip(fac, 0.0, 1.0) * (n - 1)
        # `clamp(float_to_int(f), 0, table_size - 1)`; `float_to_int` truncates and f >= 0.
        i = dr.clip(Int(f), 0, n - 1)
        t = f - Float(i)
        e0 = dr.gather(Array4f, self.table, i)
        if not self.interpolate:
            return mi.Color3f(e0[0], e0[1], e0[2]), mi.Float(e0[3])
        # `i + 1` is only read when t > 0, which cannot happen at i == n - 1 (there f is
        # exactly n - 1). Clamping the index keeps the gather in range regardless, and the
        # weight is 0 there, so this is the branch's arithmetic without the branch.
        e1 = dr.gather(Array4f, self.table, dr.minimum(i + 1, n - 1))
        w = dr.maximum(t, 0.0)
        e = (1.0 - w) * e0 + w * e1
        return mi.Color3f(e[0], e[1], e[2]), mi.Float(e[3])

    @dr.syntax
    def process_elements(self, si, active):
        fac = self.fac.eval_1(si, active)

        res_c = mi.Color3f(0)
        res_a = mi.Float(0)
        right_idx = self.get_right_stop(fac)

        if right_idx == mi.Int(0):
            res_c = self.get_element_color(right_idx)
            res_a = self.get_element_alpha(right_idx)
        elif right_idx == self.get_element_len():
            res_c = self.get_element_color(right_idx - 1)
            res_a = self.get_element_alpha(right_idx - 1)
        else:
            left_idx = right_idx - 1
            fac = (fac - self.get_element_pos(left_idx)) / (self.get_element_pos(right_idx) - self.get_element_pos(left_idx))
            res_c = (1 - fac) * self.get_element_color(left_idx) + fac * self.get_element_color(right_idx)
            res_a = (1 - fac) * self.get_element_alpha(left_idx) + fac * self.get_element_alpha(right_idx)

        return res_c, res_a

    @dr.syntax
    def get_right_stop(self, fac):
        idx = mi.Int(0)
        while (idx < self.positions.shape[0]) & (self.get_element_pos(idx) < fac):
            idx += 1

        return idx

    def resolution(self):
        return self.fac.resolution()

    def is_spatially_varying(self):
        return self.fac.is_spatially_varying()

    def to_string(self):
        if self.table is not None:
            return (f'ColorRamp[baked table, size={self.table_size}, '
                    f'interpolate={self.interpolate}, factor={self.fac}]')
        return f'ColorRamp[mode={self.mode},interpolation={self.inter},hue_interpolation={self.hue_inter},factor={self.fac}]'

mi.register_texture('color_ramp', lambda props: Ramp(props))
