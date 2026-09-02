from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
from ._base import TextureBase

# Blender's Image Texture node offers three interpolations and four extensions. Mitsuba's
# `bitmap` texture expresses NINE of those twelve combinations and no more: `Cubic` has no
# `filter_type` counterpart and `CLIP` has no `wrap_mode` counterpart. The exporter used to
# drop both fields entirely, so a CLIP texture rendered tiling and a Cubic one rendered
# bilinear -- wrong images that nothing reported. It now raises for the three it cannot
# express, and routes them HERE instead.
#
# This is Cycles' own `ImageInterpolator` from `intern/cycles/kernel/device/cpu/image.h`
# transcribed rather than approximated: the `-0.5` sample centring, the truncate-then-bias
# `frac`, the four `wrap_*` functions, the uniform cubic B-spline weights, and CLIP's
# "transparent black outside, including the taps" rule are all Cycles' arithmetic. The one
# place this deliberately DIFFERS is the vertical convention: Cycles indexes rows
# bottom-up (v = 0 is the bottom of the image), Mitsuba indexes them in file order
# (v = 0 is the top row), and the exported meshes already carry the flipped UVs -- so this
# plugin works in MITSUBA's convention throughout, exactly as `bitmap` does. That is not an
# approximation: y_m = H - y_c maps texel centres onto each other, and both the linear and
# the cubic weights are symmetric under (taps reversed, t -> 1-t), since
# u[0](t) = (1-t)^3/6 = u[3](1-t).
#
# EXTENSION and INTERPOLATION carry Blender's own enum identifiers, so the exporter passes
# `node.extension` / `node.interpolation` straight through with nothing to interpret.

EXTENSIONS = ('REPEAT', 'EXTEND', 'CLIP', 'MIRROR')
INTERPOLATIONS = ('Closest', 'Linear', 'Cubic')


def _frac(x):
    '''Cycles' `frac`: `i = float_to_int(x) - (x < 0)`, which is floor() EXCEPT at exact
    negative integers, where it returns (i-1, 1.0) instead of (i, 0.0). Transcribed rather
    than replaced by dr.floor: the two agree for every interpolated lookup (tap i+1 with
    weight 1 is tap i+1) but disagree for `Closest`, which uses the index directly.'''
    i = mi.Int32(dr.trunc(x)) - dr.select(x < 0.0, mi.Int32(1), mi.Int32(0))
    return x - mi.Float(i), i


def _wrap_periodic(x, w):
    m = x % w
    return dr.select(m < 0, m + w, m)


def _wrap_clamp(x, w):
    return dr.clip(x, 0, w - 1)


def _wrap_mirror(x, w):
    m = dr.abs(x + dr.select(x < 0, mi.Int32(1), mi.Int32(0))) % (2 * w)
    return dr.select(m >= w, 2 * w - m - 1, m)


def _cubic_weights(t):
    '''SET_CUBIC_SPLINE_WEIGHTS -- the uniform cubic B-spline basis.'''
    return (((((-1.0 / 6.0) * t + 0.5) * t - 0.5) * t + (1.0 / 6.0)),
            ((0.5 * t - 1.0) * t) * t + (2.0 / 3.0),
            ((-0.5 * t + 0.5) * t + 0.5) * t + (1.0 / 6.0),
            (1.0 / 6.0) * t * t * t)


class BlenderImage(TextureBase):
    '''
    Blender's Image Texture node, with Cycles' interpolation and extension semantics.

    This plugin is used in the Blender-Mitsuba add-on for the three
    interpolation/extension combinations `bitmap` cannot express.
    '''

    def __init__(self, props):
        TextureBase.__init__(self, props)

        self.extension = str(props.get('extension', 'REPEAT'))
        if self.extension not in EXTENSIONS:
            raise RuntimeError("blender_image: unknown extension '%s'; expected one of %s"
                               % (self.extension, ', '.join(EXTENSIONS)))
        self.interpolation = str(props.get('interpolation', 'Linear'))
        if self.interpolation not in INTERPOLATIONS:
            raise RuntimeError("blender_image: unknown interpolation '%s'; expected one of %s"
                               % (self.interpolation, ', '.join(INTERPOLATIONS)))

        # WHICH OUTPUT SOCKET OF BLENDER'S IMAGE TEXTURE NODE THIS READ IS.
        #
        # HSR: this plugin used to have no such notion and convert unconditionally to RGB,
        # which DISCARDS the alpha channel. A caller that had wired the node's Alpha socket
        # -- Blender's Principled `Alpha` input, i.e. every alpha-cutout material there is --
        # reached `eval_1`, and `eval_1` answered with the LUMINANCE OF THE COLOUR. Nothing
        # reported it: an alpha-cutout billboard rendered as a fully-formed opaque quad, and
        # a luminance is a plausible-looking mask, so the image did not obviously accuse
        # anyone. Measured on the fireplace-flame texture in Blender's 4.1 splash scene
        # (`gas+fireplace.png`, 600x315, RGBA): true alpha mean 0.31707 with 68.5% of texels
        # below 0.5, against a substituted luminance mean of 0.83261 -- 2.6x too opaque.
        #
        # Spelled as an `output` string rather than inferred from arity because arity cannot
        # carry it: `eval` (spectrum), `eval_1` (float) and `eval_3` all have to answer for
        # the SAME socket, and a caller asking for alpha as a spectrum -- which is exactly
        # what a `math` MULTIPLY scaling an emitter's radiance does, since `Math.eval` calls
        # `input_1.eval` -- is indistinguishable from one asking for colour. This mirrors
        # `color_ramp`, which already carries an `output` for the same reason.
        self.output = str(props.get('output', 'Color'))
        if self.output not in ('Color', 'Alpha'):
            raise RuntimeError("blender_image: unknown output '%s'; expected 'Color' or "
                               "'Alpha'" % self.output)

        filename = str(props.get('filename'))
        raw = bool(props.get('raw', False))
        bmp = mi.Bitmap(filename)
        import numpy as np
        if self.output == 'Alpha':
            # Alpha is never gamma-encoded, so nothing is decoded here: the convert is asked
            # to KEEP whatever gamma flag the source has, making it a channel re-layout and
            # not a colour transform. An image with no alpha is opaque, which is what Blender
            # reports for the Alpha socket of an RGB image.
            if bmp.has_alpha():
                bmp = bmp.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32,
                                  bmp.srgb_gamma())
                a = np.array(bmp, copy=True).astype(np.float32)[..., 3]
            else:
                bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32,
                                  bmp.srgb_gamma())
                a = np.ones(np.array(bmp).shape[:2], dtype=np.float32)
            # Broadcast to three channels so the interpolation, wrapping and CLIP handling
            # below are shared verbatim with the colour path rather than forked. `eval_1`
            # then returns `luminance(a, a, a)`, whose weights sum to one, so it is `a`
            # exactly; and CLIP's transparent-black outside the image is alpha 0, which is
            # what Cycles' `read_clip` gives.
            arr = np.repeat(a[..., None], 3, axis=2)
        else:
            # `raw` means "these are not colours" (normal maps, roughness, masks). Decoding
            # sRGB into them is the same silent wrong-render the exporter's colourspace
            # branch exists to prevent, so the target keeps the source's gamma flag in that
            # case and drops it otherwise -- which decodes an sRGB source and leaves a linear
            # one alone.
            target_srgb = bmp.srgb_gamma() if raw else False
            bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, target_srgb)
            arr = np.array(bmp, copy=True).astype(np.float32)   # (H, W, 3), row 0 = top
        self.height, self.width = int(arr.shape[0]), int(arr.shape[1])
        self.data = mi.Float(arr.ravel())
        self._mean = mi.Float(float(arr.mean()))

    # ------------------------------------------------------------------ lookups

    def _read(self, ix, iy, active):
        '''Cycles' `read`: no bounds check, the caller has already wrapped.'''
        return dr.gather(mi.Color3f, self.data, iy * self.width + ix, active)

    def _read_clip(self, ix, iy, active):
        '''Cycles' `read_clip`: transparent black outside the image.'''
        inside = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        return dr.select(inside, self._read(ix, iy, active & inside), mi.Color3f(0.0))

    def _wrap(self, x, w):
        if self.extension == 'REPEAT':
            return _wrap_periodic(x, w)
        if self.extension == 'EXTEND':
            return _wrap_clamp(x, w)
        if self.extension == 'MIRROR':
            return _wrap_mirror(x, w)
        return x                                   # CLIP wraps nothing; read_clip zeroes it

    def _interp_closest(self, x, y, active):
        _, ix = _frac(x)
        _, iy = _frac(y)
        if self.extension == 'CLIP':
            return self._read_clip(ix, iy, active)
        return self._read(self._wrap(ix, self.width), self._wrap(iy, self.height), active)

    def _interp_linear(self, x, y, active):
        tx, ix = _frac(x - 0.5)
        ty, iy = _frac(y - 0.5)
        if self.extension == 'CLIP':
            # Cycles early-outs the whole lookup when no tap can be inside; `read_clip`
            # already produces that value tap by tap, so the branch is redundant here.
            r = self._read_clip
            xs, ys = (ix, ix + 1), (iy, iy + 1)
        else:
            r = self._read
            xs = (self._wrap(ix, self.width), self._wrap(ix + 1, self.width))
            ys = (self._wrap(iy, self.height), self._wrap(iy + 1, self.height))
        return ((1.0 - ty) * (1.0 - tx) * r(xs[0], ys[0], active) +
                (1.0 - ty) * tx * r(xs[1], ys[0], active) +
                ty * (1.0 - tx) * r(xs[0], ys[1], active) +
                ty * tx * r(xs[1], ys[1], active))

    def _interp_cubic(self, x, y, active):
        tx, ix = _frac(x - 0.5)
        ty, iy = _frac(y - 0.5)
        if self.extension == 'CLIP':
            xs = [ix - 1, ix, ix + 1, ix + 2]
            ys = [iy - 1, iy, iy + 1, iy + 2]
        else:
            xs = [self._wrap(ix + d, self.width) for d in (-1, 0, 1, 2)]
            ys = [self._wrap(iy + d, self.height) for d in (-1, 0, 1, 2)]
        u = _cubic_weights(tx)
        v = _cubic_weights(ty)
        # Cycles reads the already-wrapped taps through `read_clip` too, which is a no-op
        # once they are in range -- so one path covers all four extensions.
        out = mi.Color3f(0.0)
        for col in range(4):
            row = mi.Color3f(0.0)
            for k in range(4):
                row += u[k] * self._read_clip(xs[k], ys[col], active)
            out += v[col] * row
        return out

    def _lookup(self, si, active):
        x = si.uv.x * self.width
        y = si.uv.y * self.height
        if self.interpolation == 'Closest':
            return self._interp_closest(x, y, active)
        if self.interpolation == 'Linear':
            return self._interp_linear(x, y, active)
        return self._interp_cubic(x, y, active)

    # ------------------------------------------------------------------ mi.Texture

    def eval_color3(self, si, active=True):
        return mi.UnpolarizedSpectrum(self._lookup(si, active))

    def eval_1(self, si, active=True):
        return mi.luminance(self._lookup(si, active))

    def eval_3(self, si, active=True):
        return self._lookup(si, active)

    def mean(self):
        return self._mean

    def is_spatially_varying(self):
        return True

    def resolution(self):
        return mi.ScalarVector2i(self.width, self.height)

    def to_string(self):
        return (f'BlenderImage[resolution=[{self.width}, {self.height}], '
                f'interpolation={self.interpolation}, extension={self.extension}, '
                f'output={self.output}]')


mi.register_field('blender_image', lambda props: BlenderImage(props))
