# Blender's colour-space conversions and its blackbody fit, transcribed from Cycles'
# `util/color.h`, `kernel/svm/math_util.h` and `kernel/tables.h`.
#
# One home, because these are shared: Hue/Saturation, the HSV/HSL blend modes of Mix Color,
# and the HSV/HSL modes of Separate and Combine Color all convert with the SAME functions in
# Blender. They used to live inside `hue_saturation`, which meant every other caller either
# imported across a plugin boundary or grew its own copy to drift.

import drjit as dr
import mitsuba as mi


def wrap01(a, b=1.0):
    """Floored modulo: the result carries `b`'s sign, so a NEGATIVE input WRAPS.

    `dr.abs` REFLECTS a negative hue instead, mapping -t to +t and so swapping green and
    blue for every hue an input pushes below zero. Measured, not deduced: with hue 0.32 on a
    red-green sweep, Cycles drew (0.76, 0, 0.79) where the reflecting form drew (0.76, 0.79, 0).
    """
    return a - b * dr.floor(a / b)


def rgb_to_hsv(rgb):
    """Blender's `rgb_to_hsv`, hue in [0, 1].

    Rewritten from a six-sector cascade of masked assignments, which relied on the sector
    predicates being mutually exclusive at the boundaries, where they are not (`R >= G >= B`
    and `R >= B >= G` are both true when G == B) -- so a later branch could silently
    overwrite an earlier one. This states the sextant once.
    """
    r, g, b = rgb.x, rgb.y, rgb.z
    mx = dr.maximum(r, dr.maximum(g, b))
    mn = dr.minimum(r, dr.minimum(g, b))
    delta = mx - mn
    safe_delta = dr.select(delta > 0.0, delta, 1.0)
    h_r = wrap01((g - b) / safe_delta, 6.0)
    h_g = (b - r) / safe_delta + 2.0
    h_b = (r - g) / safe_delta + 4.0
    h = dr.select(delta <= 0.0, 0.0,
                  dr.select(mx == r, h_r, dr.select(mx == g, h_g, h_b))) / 6.0
    s = dr.select(mx > 0.0, delta / dr.select(mx > 0.0, mx, 1.0), 0.0)
    return mi.Color3f(h, s, mx)


def hsv_to_rgb(hsv):
    """Blender's `hsv_to_rgb`: three clamped triangle waves of the hue, hue in [0, 1]."""
    h = wrap01(hsv.x, 1.0)
    s, v = hsv.y, hsv.z
    nr = dr.clip(dr.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    ng = dr.clip(2.0 - dr.abs(h * 6.0 - 2.0), 0.0, 1.0)
    nb = dr.clip(2.0 - dr.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return mi.Color3f(((nr - 1.0) * s + 1.0) * v,
                      ((ng - 1.0) * s + 1.0) * v,
                      ((nb - 1.0) * s + 1.0) * v)


def rgb_to_hsl(rgb):
    """Blender's `rgb_to_hsl`. NOT HSV with a renamed third channel: the lightness is the
    MIDPOINT of the extremes, the saturation has two branches either side of L = 0.5, and
    the L it returns is CLAMPED to 1 while the V of HSV is not."""
    r, g, b = rgb.x, rgb.y, rgb.z
    cmax = dr.maximum(r, dr.maximum(g, b))
    cmin = dr.minimum(r, dr.minimum(g, b))
    l = dr.minimum(mi.Float(1.0), (cmax + cmin) / 2.0)
    cdelta = cmax - cmin
    chroma = cmax != cmin
    safe_delta = dr.select(chroma, cdelta, 1.0)
    denom_hi = 2.0 - cmax - cmin
    denom_lo = cmax + cmin
    s = dr.select(l > 0.5,
                  cdelta / dr.select(denom_hi != 0.0, denom_hi, 1.0),
                  cdelta / dr.select(denom_lo != 0.0, denom_lo, 1.0))
    h_r = (g - b) / safe_delta + dr.select(g < b, 6.0, 0.0)
    h_g = (b - r) / safe_delta + 2.0
    h_b = (r - g) / safe_delta + 4.0
    h = dr.select(cmax == r, h_r, dr.select(cmax == g, h_g, h_b)) / 6.0
    return mi.Color3f(dr.select(chroma, h, 0.0), dr.select(chroma, s, 0.0), l)


def hsl_to_rgb(hsl):
    """Blender's `hsl_to_rgb`. The triangle waves are the same as HSV's, but they are
    centred on L and scaled by a chroma that COLLAPSES at both ends of the lightness range."""
    h, s, l = hsl.x, hsl.y, hsl.z
    nr = dr.clip(dr.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    ng = dr.clip(2.0 - dr.abs(h * 6.0 - 2.0), 0.0, 1.0)
    nb = dr.clip(2.0 - dr.abs(h * 6.0 - 4.0), 0.0, 1.0)
    chroma = (1.0 - dr.abs(2.0 * l - 1.0)) * s
    return mi.Color3f((nr - 0.5) * chroma + l,
                      (ng - 0.5) * chroma + l,
                      (nb - 0.5) * chroma + l)


# `blackbody_table_{r,g,b}` -- Cycles' piecewise fit of the Planckian locus in Rec.709.
# R and G are `a/t + b*t + c`; B is a cubic in t. The knots are temperatures, not indices.
BLACKBODY_KNOTS = (965.0, 1167.0, 1449.0, 1902.0, 3315.0, 6365.0)
BLACKBODY_TABLE_R = (
    (1.61919106e+03, -2.05010916e-03, 5.02995757e+00),
    (2.48845471e+03, -1.11330907e-03, 3.22621544e+00),
    (3.34143193e+03, -4.86551192e-04, 1.76486769e+00),
    (4.09461742e+03, -1.27446582e-04, 7.25731635e-01),
    (4.67028036e+03, 2.91258199e-05, 1.26703442e-01),
    (4.59509185e+03, 2.87495649e-05, 1.50345020e-01),
    (3.78717450e+03, 9.35907826e-06, 3.99075871e-01))
BLACKBODY_TABLE_G = (
    (-4.88999748e+02, 6.04330754e-04, -7.55807526e-02),
    (-7.55994277e+02, 3.16730098e-04, 4.78306139e-01),
    (-1.02363977e+03, 1.20223470e-04, 9.36662319e-01),
    (-1.26571316e+03, 4.87340896e-06, 1.27054498e+00),
    (-1.42529332e+03, -4.01150431e-05, 1.43972784e+00),
    (-1.17554822e+03, -2.16378048e-05, 1.30408023e+00),
    (-5.00799571e+02, -4.59832026e-06, 1.09098763e+00))
BLACKBODY_TABLE_B = (
    (5.96945309e-11, -4.85742887e-08, -9.70622247e-05, -4.07936148e-03),
    (2.40430366e-11, 5.55021075e-08, -1.98503712e-04, 2.89312858e-02),
    (-1.40949732e-11, 1.89878968e-07, -3.56632824e-04, 9.10767778e-02),
    (-3.61460868e-11, 2.84822009e-07, -4.93211319e-04, 1.56723440e-01),
    (-1.97075738e-11, 1.75359352e-07, -2.50542825e-04, -2.22783266e-02),
    (-1.61997957e-13, -1.64216008e-08, 3.86216271e-04, -7.38077418e-01),
    (6.72650283e-13, -2.73078809e-08, 4.24098264e-04, -7.52335691e-01))

BLACKBODY_ABOVE = (0.8262954810464208, 0.9945080501520986, 1.566307710274283)
BLACKBODY_BELOW = (5.413294490189271, -0.20319390035873933, -0.0822535242887164)


def blackbody_rec709(t):
    """Cycles' `svm_math_blackbody_color_rec709`, clamped to non-negative as the node does.

    The fit is deliberately allowed to go NEGATIVE inside its range -- that is how it
    represents a gamut wider than Rec.709 -- so the clamp is part of the node, not a guard
    bolted on here.
    """
    t_inv = dr.rcp(dr.maximum(t, 1.0))
    r = mi.Float(BLACKBODY_TABLE_R[0][0]) * t_inv + BLACKBODY_TABLE_R[0][1] * t + BLACKBODY_TABLE_R[0][2]
    g = mi.Float(BLACKBODY_TABLE_G[0][0]) * t_inv + BLACKBODY_TABLE_G[0][1] * t + BLACKBODY_TABLE_G[0][2]
    bb = BLACKBODY_TABLE_B[0]
    b = ((bb[0] * t + bb[1]) * t + bb[2]) * t + bb[3]
    for i, knot in enumerate(BLACKBODY_KNOTS, start=1):
        use = t >= knot
        rr, gg, bb = BLACKBODY_TABLE_R[i], BLACKBODY_TABLE_G[i], BLACKBODY_TABLE_B[i]
        r = dr.select(use, rr[0] * t_inv + rr[1] * t + rr[2], r)
        g = dr.select(use, gg[0] * t_inv + gg[1] * t + gg[2], g)
        b = dr.select(use, ((bb[0] * t + bb[1]) * t + bb[2]) * t + bb[3], b)
    r = dr.select(t >= 12000.0, BLACKBODY_ABOVE[0], dr.select(t < 800.0, BLACKBODY_BELOW[0], r))
    g = dr.select(t >= 12000.0, BLACKBODY_ABOVE[1], dr.select(t < 800.0, BLACKBODY_BELOW[1], g))
    b = dr.select(t >= 12000.0, BLACKBODY_ABOVE[2], dr.select(t < 800.0, BLACKBODY_BELOW[2], b))
    return mi.Color3f(dr.maximum(r, 0.0), dr.maximum(g, 0.0), dr.maximum(b, 0.0))
