# Cycles' integer and float hashes, from `intern/cycles/util/hash.h`.
#
# Jenkins lookup3 with the key LENGTH mixed into the seed, which is why `hash_uint2(x, y)`
# is not `hash_uint3(x, y, 0)`. Everything Blender calls "noise" or "random" -- the Noise
# Texture's Perlin lattice, White Noise, the per-object and per-instance random values --
# resolves to these, so getting them exactly right once is what makes any of them match.
#
# The float variants hash the IEEE BIT PATTERN of their arguments, not their values. Cycles'
# own comment on why a faster modern hash is not substituted: "the current behavior has to be
# kept to match what is possible in OSL (OSL lacks bit casts and unsigned integers)."

import struct

import drjit as dr
import mitsuba as mi

UINT_MAX_F = 1.0 / float(0xFFFFFFFF)


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


def seed(n):
    """`0xdeadbeef + (n << 2) + 13` -- the key length is part of the seed."""
    return (0xdeadbeef + (n << 2) + 13) & 0xffffffff


def hash_uint(kx):
    s = seed(1)
    return _final(mi.UInt32(s) + kx, mi.UInt32(s), mi.UInt32(s))


def hash_uint2(kx, ky):
    s = seed(2)
    return _final(mi.UInt32(s) + kx, mi.UInt32(s) + ky, mi.UInt32(s))


def hash_uint3(kx, ky, kz):
    s = seed(3)
    return _final(mi.UInt32(s) + kx, mi.UInt32(s) + ky, mi.UInt32(s) + kz)


def hash_uint4(kx, ky, kz, kw):
    s = seed(4)
    a, b, c = _mix(mi.UInt32(s) + kx, mi.UInt32(s) + ky, mi.UInt32(s) + kz)
    return _final(a + kw, b, c)


def as_uint(f):
    """`__float_as_uint`: the BIT PATTERN, not the value."""
    return dr.reinterpret_array(mi.UInt32, mi.Float(f))


def _to_float(u):
    return mi.Float(u) * UINT_MAX_F


def hash_float_to_float(k):
    return _to_float(hash_uint(as_uint(k)))


def hash_float2_to_float(x, y):
    return _to_float(hash_uint2(as_uint(x), as_uint(y)))


def hash_float3_to_float(x, y, z):
    return _to_float(hash_uint3(as_uint(x), as_uint(y), as_uint(z)))


def hash_float4_to_float(x, y, z, w):
    return _to_float(hash_uint4(as_uint(x), as_uint(y), as_uint(z), as_uint(w)))


def hash_float_to_float3(k):
    return mi.Color3f(hash_float_to_float(k),
                      hash_float2_to_float(k, 1.0),
                      hash_float2_to_float(k, 2.0))


def hash_float2_to_float3(x, y):
    return mi.Color3f(hash_float2_to_float(x, y),
                      hash_float3_to_float(x, y, 1.0),
                      hash_float3_to_float(x, y, 2.0))


def hash_float3_to_float3(x, y, z):
    return mi.Color3f(hash_float3_to_float(x, y, z),
                      hash_float4_to_float(x, y, z, 1.0),
                      hash_float4_to_float(x, y, z, 2.0))


def hash_float4_to_float3(x, y, z, w):
    # The component ROTATIONS are Cycles'; a different permutation gives a different -- and
    # equally noise-looking -- picture.
    return mi.Color3f(hash_float4_to_float(x, y, z, w),
                      hash_float4_to_float(z, x, w, y),
                      hash_float4_to_float(w, z, y, x))


# ---------------------------------------------------------------------------------------
# The same lookup3 on plain Python ints, for constants that can be folded at import time.
# ---------------------------------------------------------------------------------------

_M = 0xffffffff


def _py_rot(x, k):
    return ((x << k) | (x >> (32 - k))) & _M


def _py_final(a, b, c):
    c = (c ^ b) & _M; c = (c - _py_rot(b, 14)) & _M
    a = (a ^ c) & _M; a = (a - _py_rot(c, 11)) & _M
    b = (b ^ a) & _M; b = (b - _py_rot(a, 25)) & _M
    c = (c ^ b) & _M; c = (c - _py_rot(b, 16)) & _M
    a = (a ^ c) & _M; a = (a - _py_rot(c, 4)) & _M
    b = (b ^ a) & _M; b = (b - _py_rot(a, 14)) & _M
    c = (c ^ b) & _M; c = (c - _py_rot(b, 24)) & _M
    return c


def py_float_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def py_hash_float_to_float(k):
    return _py_final((seed(1) + py_float_bits(float(k))) & _M, seed(1), seed(1)) * UINT_MAX_F


def py_hash_float2_to_float(kx, ky):
    return _py_final((seed(2) + py_float_bits(float(kx))) & _M,
                     (seed(2) + py_float_bits(float(ky))) & _M, seed(2)) * UINT_MAX_F
