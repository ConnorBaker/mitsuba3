# Cycles' `safe_*` / `compatible_*` scalar helpers, transcribed from
# `intern/cycles/util/math_base.h` and used by every plugin that has to agree with a Blender
# node numerically.
#
# These exist because the difference between `a / b` and Cycles' `safe_divide(a, b)` is not
# an edge case in a renderer: `b == 0` is an ordinary value of an ordinary socket, and the
# two answers are `inf` and `0`. Every one of the wrappers below marks a place where the
# obvious spelling and Blender's spelling disagree on inputs a texture reaches constantly:
#
#   dr.power(-2, 0.5)   -> NaN            safe_pow      -> 0
#   dr.sign(0.0)        -> 1              compat_sign   -> 0
#   dr.round(0.5)       -> 0 (half-even)  Blender       -> 1 (floor(a + 0.5))
#   dr.rcp(dr.sqrt(0))  -> inf            inverse_sqrt  -> 0
#   log(a)/log(b), a<=0 -> NaN/-inf       safe_log      -> 0
#
# Componentwise by construction, so the same function serves the scalar Math node and the
# Vector Math node without a second implementation to drift.

import drjit as dr

# `FLT_EPSILON`, which is what Cycles' COMPARE uses as its floor -- not 1e-5.
FLT_EPSILON = 1.1920928955078125e-07


def fract(x):
    return x - dr.floor(x)


def safe_divide(a, b):
    return dr.select(b != 0.0, a / dr.select(b != 0.0, b, 1.0), 0.0)


def safe_sqrt(a):
    return dr.sqrt(dr.maximum(a, 0.0))


def inverse_sqrt(a):
    return dr.select(a > 0.0, dr.rcp(dr.sqrt(dr.maximum(a, FLT_EPSILON))), 0.0)


def compatible_pow(x, y):
    # Cycles' GPU path: negative bases are resolved by the PARITY of the exponent rather
    # than handed to `powf`, which returns NaN for them.
    mag = dr.power(dr.abs(x), y)
    neg_odd = (x < 0.0) & (fract(dr.abs(y) * 0.5) != 0.0)
    r = dr.select(neg_odd, -mag, mag)
    r = dr.select(x == 0.0, 0.0, r)
    return dr.select(y == 0.0, 1.0, r)


def safe_pow(a, b):
    # A negative base raised to a NON-INTEGER exponent is 0 in Blender, not NaN.
    return dr.select((a < 0.0) & (b != dr.trunc(b)), 0.0, compatible_pow(a, b))


def safe_log(a, b):
    ok = (a > 0.0) & (b > 0.0)
    return dr.select(ok, safe_divide(dr.log(dr.select(ok, a, 1.0)),
                                     dr.log(dr.select(ok, b, 2.0))), 0.0)


def safe_modulo(a, b):
    # C's `fmodf`: truncated, so the result carries the sign of `a`.
    return dr.select(b != 0.0, a - dr.trunc(safe_divide(a, b)) * b, 0.0)


def safe_floored_modulo(a, b):
    return dr.select(b != 0.0, a - dr.floor(safe_divide(a, b)) * b, 0.0)


def compatible_sign(a):
    return dr.select(a == 0.0, 0.0, dr.select(a > 0.0, 1.0, -1.0))


def blender_round(a):
    # `floorf(a + 0.5f)`. `dr.round` is round-half-to-EVEN, which disagrees at every .5 --
    # and a .5 is exactly what a texture coordinate lands on.
    return dr.floor(a + 0.5)


def blender_trunc(a):
    return dr.select(a >= 0.0, dr.floor(a), dr.ceil(a))


def safe_asin(a):
    return dr.asin(dr.clip(a, -1.0, 1.0))


def safe_acos(a):
    return dr.acos(dr.clip(a, -1.0, 1.0))


def compatible_atan2(y, x):
    both_zero = (x == 0.0) & (y == 0.0)
    return dr.select(both_zero, 0.0, dr.atan2(dr.select(both_zero, 1.0, y), x))


def wrap(value, hi, lo):
    # Cycles' `wrapf(value, max, min)`. There is no `value == max` special case in Cycles;
    # the GLSL viewport shader has one, and copying THAT here would make the offline render
    # disagree with the offline render.
    rng = hi - lo
    safe_rng = dr.select(rng != 0.0, rng, 1.0)
    return dr.select(rng != 0.0, value - rng * dr.floor((value - lo) / safe_rng), lo)


def pingpong(a, b):
    d = b * 2.0
    return dr.select(b != 0.0, dr.abs(fract(safe_divide(a - b, d)) * d - b), 0.0)


def smooth_min(a, b, k):
    safe_k = dr.select(k != 0.0, k, 1.0)
    h = dr.maximum(safe_k - dr.abs(a - b), 0.0) / safe_k
    return dr.select(k != 0.0,
                     dr.minimum(a, b) - h * h * h * safe_k * (1.0 / 6.0),
                     dr.minimum(a, b))


def compare(a, b, c):
    return dr.select((a == b) | (dr.abs(a - b) <= dr.maximum(c, FLT_EPSILON)), 1.0, 0.0)
