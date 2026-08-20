# Blender's Noise Texture node, reproduced from Cycles' own source rather than approximated.
#
# Every constant, every hash, every octave-weighting rule below is transcribed from
# `intern/cycles/kernel/svm/{noise,fractal_noise,noisetex}.h` and `intern/cycles/util/hash.h`
# at the Blender tag this repository renders against. That is deliberate: Perlin noise is
# not a family of interchangeable functions but a specific one, pinned by a specific integer
# hash and a specific gradient table, and "a Perlin noise" that is not Cycles' Perlin noise
# produces a picture that looks plausible and matches nothing.
#
# The plugin this supersedes for Blender export (`noise`) is 2D Perlin evaluated on `si.uv`
# at a fixed 128-pixel resolution, with `roughness`, `lacunarity` and `distortion` accepted
# and never read. It is left in place for its own users; it is not what a Noise Texture node
# means.

import drjit as dr
import mitsuba as mi

import struct

NOISE_TYPES = ('FBM', 'MULTIFRACTAL', 'HETERO_TERRAIN',
               'RIDGED_MULTIFRACTAL', 'HYBRID_MULTIFRACTAL')

# Cycles clamps `detail` into this range before use, so 15 octaves plus the fractional one
# is the whole reachable space -- the unrolled loop below is not a truncation.
MAX_DETAIL = 15.0

# `noise_scale{1,2,3,4}`. Cycles' comment: "The scale values were computed experimentally by
# the OSL developers." They remap Perlin's raw output to [-1, 1] and they are NOT 1.
NOISE_SCALE = (0.2500, 0.6616, 0.9820, 0.8344)

# Perlin is repeated every 100000 on each axis to keep float32 coordinates well-conditioned.
NOISE_REPEAT = 100000.0
NOISE_PRECISION_THRESHOLD = 1000000.0


# ---------------------------------------------------------------------------------------
# Jenkins lookup3, as Cycles spells it. The hash is the identity of the noise: change it and
# every value changes while the picture still looks like noise.
# ---------------------------------------------------------------------------------------

def _rot(x, k):
    return (x << k) | (x >> (32 - k))


def _final(a, b, c):
    c ^= b; c -= _rot(b, 14)
    a ^= c; a -= _rot(c, 11)
    b ^= a; b -= _rot(a, 25)
    c ^= b; c -= _rot(b, 16)
    a ^= c; a -= _rot(c, 4)
    b ^= a; b -= _rot(a, 14)
    c ^= b; c -= _rot(b, 24)
    return c


def _mix(a, b, c):
    a -= c; a ^= _rot(c, 4);  c += b
    b -= a; b ^= _rot(a, 6);  a += c
    c -= b; c ^= _rot(b, 8);  b += a
    a -= c; a ^= _rot(c, 16); c += b
    b -= a; b ^= _rot(a, 19); a += c
    c -= b; c ^= _rot(b, 4);  b += a
    return a, b, c


def _seed(n):
    # `a = b = c = 0xdeadbeef + (n << 2) + 13` -- the length is mixed into the seed, which is
    # why hash_uint2(x, y) is not hash_uint3(x, y, 0).
    return (0xdeadbeef + (n << 2) + 13) & 0xffffffff


def hash_uint(kx):
    s = _seed(1)
    a, b, c = mi.UInt32(s) + kx, mi.UInt32(s), mi.UInt32(s)
    return _final(a, b, c)


def hash_uint2(kx, ky):
    s = _seed(2)
    a, b, c = mi.UInt32(s) + kx, mi.UInt32(s) + ky, mi.UInt32(s)
    return _final(a, b, c)


def hash_uint3(kx, ky, kz):
    s = _seed(3)
    a, b, c = mi.UInt32(s) + kx, mi.UInt32(s) + ky, mi.UInt32(s) + kz
    return _final(a, b, c)


def hash_uint4(kx, ky, kz, kw):
    s = _seed(4)
    a, b, c = _mix(mi.UInt32(s) + kx, mi.UInt32(s) + ky, mi.UInt32(s) + kz)
    return _final(a + kw, b, c)


def _py_hash_uint2(kx, ky):
    """The same hash on plain Python ints, for the compile-time distortion offsets."""
    M = 0xffffffff

    def rot(x, k):
        return ((x << k) | (x >> (32 - k))) & M

    a = (_seed(2) + kx) & M
    b = (_seed(2) + ky) & M
    c = _seed(2)
    c = (c ^ b) & M; c = (c - rot(b, 14)) & M
    a = (a ^ c) & M; a = (a - rot(c, 11)) & M
    b = (b ^ a) & M; b = (b - rot(a, 25)) & M
    c = (c ^ b) & M; c = (c - rot(b, 16)) & M
    a = (a ^ c) & M; a = (a - rot(c, 4)) & M
    b = (b ^ a) & M; b = (b - rot(a, 14)) & M
    c = (c ^ b) & M; c = (c - rot(b, 24)) & M
    return c


def _float_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def _random_offset_component(seed, k):
    # `100.0f + hash_float2_to_float(make_float2(seed, k)) * 100.0f`, and
    # `hash_float2_to_float` hashes the IEEE BIT PATTERNS of the two floats, not their values.
    h = _py_hash_uint2(_float_bits(float(seed)), _float_bits(float(k)))
    return 100.0 + (h * (1.0 / float(0xFFFFFFFF))) * 100.0


def random_offset(seed, n):
    """`random_float{n}_offset(seed)` -- a constant, so it is evaluated once at import."""
    if n == 1:
        # The 1D variant uses `hash_float_to_float`, i.e. the ONE-argument hash, whose seed
        # constant differs. It is not `random_offset(seed, 2)[0]`.
        M = 0xffffffff

        def rot(x, k):
            return ((x << k) | (x >> (32 - k))) & M
        a = (_seed(1) + _float_bits(float(seed))) & M
        b = c = _seed(1)
        c = (c ^ b) & M; c = (c - rot(b, 14)) & M
        a = (a ^ c) & M; a = (a - rot(c, 11)) & M
        b = (b ^ a) & M; b = (b - rot(a, 25)) & M
        c = (c ^ b) & M; c = (c - rot(b, 16)) & M
        a = (a ^ c) & M; a = (a - rot(c, 4)) & M
        b = (b ^ a) & M; b = (b - rot(a, 14)) & M
        c = (c ^ b) & M; c = (c - rot(b, 24)) & M
        return (100.0 + (c * (1.0 / float(M))) * 100.0,)
    return tuple(_random_offset_component(seed, k) for k in range(n))


# ---------------------------------------------------------------------------------------
# Perlin
# ---------------------------------------------------------------------------------------

def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _floorfrac(x):
    f = dr.floor(x)
    return x - f, dr.reinterpret_array(mi.UInt32, mi.Int32(f))


def _negate_if(val, cond_bits):
    return dr.select(cond_bits != 0, -val, val)


def _grad1(h32, x):
    h = h32 & 15
    g = mi.Float(1 + (h & 7))
    return _negate_if(g, h & 8) * x


def _grad2(h32, x, y):
    h = h32 & 7
    lt4 = h < 4
    u = dr.select(lt4, x, y)
    v = 2.0 * dr.select(lt4, y, x)
    return _negate_if(u, h & 1) + _negate_if(v, h & 2)


def _grad3(h32, x, y, z):
    h = h32 & 15
    u = dr.select(h < 8, x, y)
    vt = dr.select((h == 12) | (h == 14), x, z)
    v = dr.select(h < 4, y, vt)
    return _negate_if(u, h & 1) + _negate_if(v, h & 2)


def _grad4(h32, x, y, z, w):
    h = h32 & 31
    u = dr.select(h < 24, x, y)
    v = dr.select(h < 16, y, z)
    s = dr.select(h < 8, z, w)
    return _negate_if(u, h & 1) + _negate_if(v, h & 2) + _negate_if(s, h & 4)


def _lerp(a, b, t):
    return a + t * (b - a)


def perlin_1d(x):
    fx, X = _floorfrac(x)
    return _lerp(_grad1(hash_uint(X), fx),
                 _grad1(hash_uint(X + 1), fx - 1.0), _fade(fx))


def perlin_2d(x, y):
    fx, X = _floorfrac(x)
    fy, Y = _floorfrac(y)
    u, v = _fade(fx), _fade(fy)
    g = lambda i, j: _grad2(hash_uint2(X + i, Y + j), fx - i, fy - j)
    x1 = 1.0 - u
    return (1.0 - v) * (g(0, 0) * x1 + g(1, 0) * u) + v * (g(0, 1) * x1 + g(1, 1) * u)


def _tri_mix(v, x, y, z):
    x1, y1, z1 = 1.0 - x, 1.0 - y, 1.0 - z
    return (z1 * (y1 * (v[0] * x1 + v[1] * x) + y * (v[2] * x1 + v[3] * x)) +
            z * (y1 * (v[4] * x1 + v[5] * x) + y * (v[6] * x1 + v[7] * x)))


def perlin_3d(x, y, z):
    fx, X = _floorfrac(x)
    fy, Y = _floorfrac(y)
    fz, Z = _floorfrac(z)
    g = lambda i, j, k: _grad3(hash_uint3(X + i, Y + j, Z + k), fx - i, fy - j, fz - k)
    corners = [g(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    return _tri_mix(corners, _fade(fx), _fade(fy), _fade(fz))


def perlin_4d(x, y, z, w):
    fx, X = _floorfrac(x)
    fy, Y = _floorfrac(y)
    fz, Z = _floorfrac(z)
    fw, W = _floorfrac(w)
    g = lambda i, j, k, l: _grad4(hash_uint4(X + i, Y + j, Z + k, W + l),
                                  fx - i, fy - j, fz - k, fw - l)
    lo = [g(i, j, k, 0) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    hi = [g(i, j, k, 1) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    u, v, t = _fade(fx), _fade(fy), _fade(fz)
    return _lerp(_tri_mix(lo, u, v, t), _tri_mix(hi, u, v, t), _fade(fw))


def _c_fmod(a, b):
    # C's fmod truncates toward zero; dr.floor would give the Python `%` instead, which
    # differs in sign for negative coordinates -- and half of every texture is negative.
    return a - b * dr.trunc(a / b)


def _precision_correct(c):
    return _c_fmod(c, NOISE_REPEAT) + 0.5 * dr.select(dr.abs(c) >= NOISE_PRECISION_THRESHOLD,
                                                      1.0, 0.0)


def snoise(p, dims):
    if dims == 1:
        return NOISE_SCALE[0] * perlin_1d(_precision_correct(p[0]))
    q = [_precision_correct(c) for c in p]
    if dims == 2:
        return NOISE_SCALE[1] * perlin_2d(q[0], q[1])
    if dims == 3:
        return NOISE_SCALE[2] * perlin_3d(q[0], q[1], q[2])
    return NOISE_SCALE[3] * perlin_4d(q[0], q[1], q[2], q[3])


# ---------------------------------------------------------------------------------------
# The five fractal variants.
#
# Cycles runs `for (i = 0; i <= float_to_int(detail); i++)`, a data-dependent trip count.
# Dr.Jit traces one kernel, so the loop is unrolled to its clamped maximum and each octave
# is MASKED by `i <= floor(detail)`. The accumulators advance only on active octaves, so
# after the loop `amp` and `fscale` hold exactly the values Cycles' loop would have left --
# which is what the fractional-octave tail then consumes. When `detail` is a constant the
# unroll is cut to the octaves that can actually be active; that is a compile-time saving,
# not a different formula.
# ---------------------------------------------------------------------------------------

def _scaled(p, s):
    return [c * s for c in p]


def _shift(p, offs):
    return [c + o for c, o in zip(p, offs)]


def fractal_fbm(p, dims, detail, roughness, lacunarity, normalize, n_oct):
    fscale, amp = mi.Float(1.0), mi.Float(1.0)
    maxamp, total = mi.Float(0.0), mi.Float(0.0)
    fdetail = dr.floor(detail)
    for i in range(n_oct):
        act = mi.Float(i) <= fdetail
        t = snoise(_scaled(p, fscale), dims)
        total += dr.select(act, t * amp, 0.0)
        maxamp += dr.select(act, amp, 0.0)
        amp = dr.select(act, amp * roughness, amp)
        fscale = dr.select(act, fscale * lacunarity, fscale)
    rmd = detail - fdetail
    t = snoise(_scaled(p, fscale), dims)
    total2 = total + t * amp
    if normalize:
        a = 0.5 * total / maxamp + 0.5
        b = 0.5 * total2 / (maxamp + amp) + 0.5
    else:
        a, b = total, total2
    return dr.select(rmd != 0.0, _lerp(a, b, rmd), a)


def fractal_multi(p, dims, detail, roughness, lacunarity, n_oct):
    value, pwr = mi.Float(1.0), mi.Float(1.0)
    q, fdetail = list(p), dr.floor(detail)
    for i in range(n_oct):
        act = mi.Float(i) <= fdetail
        f = pwr * snoise(q, dims) + 1.0
        value *= dr.select(act, f, 1.0)
        pwr = dr.select(act, pwr * roughness, pwr)
        q = [dr.select(act, c * lacunarity, c) for c in q]
    rmd = detail - fdetail
    tail = rmd * pwr * snoise(q, dims) + 1.0
    return value * dr.select(rmd != 0.0, tail, 1.0)


def fractal_hetero_terrain(p, dims, detail, roughness, lacunarity, offset, n_oct):
    pwr = mi.Float(roughness)
    value = offset + snoise(p, dims)
    q = _scaled(p, lacunarity)
    fdetail = dr.floor(detail)
    for i in range(1, n_oct):
        act = mi.Float(i) <= fdetail
        inc = (snoise(q, dims) + offset) * pwr * value
        value += dr.select(act, inc, 0.0)
        pwr = dr.select(act, pwr * roughness, pwr)
        q = [dr.select(act, c * lacunarity, c) for c in q]
    rmd = detail - fdetail
    inc = (snoise(q, dims) + offset) * pwr * value
    return value + dr.select(rmd != 0.0, rmd * inc, 0.0)


def fractal_hybrid_multi(p, dims, detail, roughness, lacunarity, offset, gain, n_oct):
    pwr, value, weight = mi.Float(1.0), mi.Float(0.0), mi.Float(1.0)
    q, fdetail = list(p), dr.floor(detail)
    for i in range(n_oct):
        # Cycles' loop carries a SECOND exit condition, `weight > 0.001f`, tested before the
        # octave runs. Masked rather than branched, and it is sticky: once a lane's weight
        # falls through the floor it must stay inactive for every later octave, which a
        # per-octave test of the CURRENT weight would not give.
        act = (mi.Float(i) <= fdetail) & (weight > 0.001)
        w = dr.minimum(weight, 1.0)
        signal = (snoise(q, dims) + offset) * pwr
        pwr = dr.select(act, pwr * roughness, pwr)
        value += dr.select(act, w * signal, 0.0)
        weight = dr.select(act, w * gain * signal, weight)
        q = [dr.select(act, c * lacunarity, c) for c in q]
    rmd = detail - fdetail
    w = dr.minimum(weight, 1.0)
    signal = (snoise(q, dims) + offset) * pwr
    tail_act = (rmd != 0.0) & (weight > 0.001)
    return value + dr.select(tail_act, rmd * w * signal, 0.0)


def fractal_ridged_multi(p, dims, detail, roughness, lacunarity, offset, gain, n_oct):
    pwr = mi.Float(roughness)
    signal = offset - dr.abs(snoise(p, dims))
    signal = signal * signal
    value = mi.Float(signal)
    q, fdetail = _scaled(p, lacunarity), dr.floor(detail)
    for i in range(1, n_oct):
        act = mi.Float(i) <= fdetail
        weight = dr.clip(signal * gain, 0.0, 1.0)
        s = offset - dr.abs(snoise(q, dims))
        s = s * s * weight
        value += dr.select(act, s * pwr, 0.0)
        signal = dr.select(act, s, signal)
        pwr = dr.select(act, pwr * roughness, pwr)
        q = [dr.select(act, c * lacunarity, c) for c in q]
    # Ridged has NO fractional-octave tail in Cycles -- `detail` is effectively floored here,
    # and that asymmetry is the source's, not a transcription slip.
    return value


class NoiseTexture(mi.Texture):
    """Blender's Noise Texture node."""

    def __init__(self, props):
        mi.Texture.__init__(self, props)

        self.dimensions = int(props.get('dimensions', 3))
        if self.dimensions not in (1, 2, 3, 4):
            raise Exception("noise_texture: `dimensions` must be 1, 2, 3 or 4, got %d"
                            % self.dimensions)

        self.noise_type = str(props.get('noise_type', 'FBM'))
        if self.noise_type not in NOISE_TYPES:
            raise Exception("noise_texture: unsupported `noise_type` '%s'; Cycles has %s"
                            % (self.noise_type, ', '.join(NOISE_TYPES)))

        self.normalize = bool(props.get('normalize', True))
        self.channel = str(props.get('channel', 'fac')).lower()
        if self.channel not in ('fac', 'color'):
            raise Exception("noise_texture: `channel` must be 'fac' or 'color', got '%s'"
                            % self.channel)

        self.vector = None
        if 'vector' in [str(k) for k in props.property_names()]:
            self.vector = props.get_unbounded_texture('vector', 0.0)

        self.scale = props.get_unbounded_texture('scale', 5.0)
        self.detail = props.get_unbounded_texture('detail', 2.0)
        self.roughness = props.get_unbounded_texture('roughness', 0.5)
        self.lacunarity = props.get_unbounded_texture('lacunarity', 2.0)
        self.offset = props.get_unbounded_texture('offset', 0.0)
        self.gain = props.get_unbounded_texture('gain', 1.0)
        self.distortion = props.get_unbounded_texture('distortion', 0.0)
        self.w = props.get_unbounded_texture('w', 0.0)

        # A constant `detail` bounds the unroll. `detail_const` is None when the socket is
        # driven by another node, and then every reachable octave is emitted.
        try:
            raw = props.get('detail', 2.0)
        except Exception:
            raw = None
        self.detail_const = float(raw) if isinstance(raw, (int, float)) else None
        if self.detail_const is None:
            self.n_octaves = int(MAX_DETAIL) + 1
        else:
            self.n_octaves = int(min(max(self.detail_const, 0.0), MAX_DETAIL)) + 1

    def _coords(self, si, active):
        scale = self.scale.eval_1(si, active)
        if self.dimensions == 1:
            return [self.w.eval_1(si, active) * scale]
        if self.vector is not None:
            v = self.vector.eval_3(si, active)
            v = mi.Vector3f(v.x, v.y, v.z)
        else:
            # The node's own default when nothing is plugged in is Generated coordinates;
            # on an exported mesh the exporter is expected to supply them explicitly, so an
            # unconnected Vector here means UV.
            v = mi.Vector3f(si.uv.x, si.uv.y, 0.0)
        p = [v.x * scale, v.y * scale, v.z * scale]
        if self.dimensions == 2:
            return p[:2]
        if self.dimensions == 3:
            return p
        return p + [self.w.eval_1(si, active) * scale]

    def _select(self, p, detail, roughness, lacunarity, offset, gain):
        d, n = self.dimensions, self.n_octaves
        if self.noise_type == 'FBM':
            return fractal_fbm(p, d, detail, roughness, lacunarity, self.normalize, n)
        if self.noise_type == 'MULTIFRACTAL':
            return fractal_multi(p, d, detail, roughness, lacunarity, n)
        if self.noise_type == 'HETERO_TERRAIN':
            return fractal_hetero_terrain(p, d, detail, roughness, lacunarity, offset, n)
        if self.noise_type == 'HYBRID_MULTIFRACTAL':
            return fractal_hybrid_multi(p, d, detail, roughness, lacunarity, offset, gain, n)
        return fractal_ridged_multi(p, d, detail, roughness, lacunarity, offset, gain, n)

    def _evaluate(self, si, active, want_color):
        p = self._coords(si, active)
        d = self.dimensions
        detail = dr.clip(self.detail.eval_1(si, active), 0.0, MAX_DETAIL)
        roughness = dr.maximum(self.roughness.eval_1(si, active), 0.0)
        lacunarity = self.lacunarity.eval_1(si, active)
        offset = self.offset.eval_1(si, active)
        gain = self.gain.eval_1(si, active)
        distortion = self.distortion.eval_1(si, active)

        # Cycles guards this with `if (distortion != 0.0f)`, i.e. the whole displacement is
        # skipped -- not multiplied by zero. Multiplying by zero happens to agree here
        # because each component is `snoise(...) * distortion`, so the mask is a cost
        # question rather than a correctness one.
        dist_p = [c + snoise(_shift(p, random_offset(k, d)), d) * distortion
                  for k, c in enumerate(p)]
        p = [dr.select(distortion != 0.0, a, b) for a, b in zip(dist_p, p)]

        value = self._select(p, detail, roughness, lacunarity, offset, gain)
        if not want_color:
            return value, None
        g = self._select(_shift(p, random_offset(d, d)),
                         detail, roughness, lacunarity, offset, gain)
        b = self._select(_shift(p, random_offset(d + 1, d)),
                         detail, roughness, lacunarity, offset, gain)
        return value, mi.Color3f(value, g, b)

    def eval(self, si, active=True):
        return self.eval_3(si, active)

    def eval_1(self, si, active=True):
        if self.channel == 'color':
            _, c = self._evaluate(si, active, True)
            return (c.x + c.y + c.z) / 3.0
        v, _ = self._evaluate(si, active, False)
        return v

    def eval_3(self, si, active=True):
        if self.channel == 'color':
            _, c = self._evaluate(si, active, True)
            return c
        v, _ = self._evaluate(si, active, False)
        return mi.Color3f(v)

    def mean(self):
        return 0.5

    def traverse(self, cb):
        for name in ('scale', 'detail', 'roughness', 'lacunarity',
                     'offset', 'gain', 'distortion', 'w'):
            cb.put(name, getattr(self, name), mi.ParamFlags.Differentiable)
        if self.vector is not None:
            cb.put('vector', self.vector, mi.ParamFlags.Differentiable)

    def to_string(self):
        return ('NoiseTexture[dimensions=%d, noise_type=%s, normalize=%s, channel=%s, '
                'octaves=%d]' % (self.dimensions, self.noise_type, self.normalize,
                                 self.channel, self.n_octaves))


mi.register_texture('noise_texture', lambda props: NoiseTexture(props))
