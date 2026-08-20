from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi


def euler_xyz_to_mat3(rotation):
    '''Blender's `euler_to_mat3` for the default XYZ order: R = Rz @ Ry @ Rx.

    Returned as three ROW vectors, so `transform` is three dot products and `transform_t`
    (the transpose, which the TEXTURE mode needs) is a column combination -- no matrix type
    and no inverse, since a rotation's inverse IS its transpose.
    '''
    cx, cy, cz = dr.cos(rotation.x), dr.cos(rotation.y), dr.cos(rotation.z)
    sx, sy, sz = dr.sin(rotation.x), dr.sin(rotation.y), dr.sin(rotation.z)
    r0 = mi.Vector3f(cz * cy, cz * sy * sx - sz * cx, sz * sx + cz * sy * cx)
    r1 = mi.Vector3f(sz * cy, cz * cx + sz * sy * sx, sz * sy * cx - cz * sx)
    r2 = mi.Vector3f(-sy, cy * sx, cy * cx)
    return r0, r1, r2


class Mapping(mi.Texture):
    '''
    Mapping Blender shader node: an affine transform of a coordinate vector.

    WHY THIS MATTERS MORE THAN IT LOOKS. The Blender exporter had no counterpart for this
    node and did not fail on it either -- it never read an image texture's `Vector` input at
    all, so a Mapping node feeding a texture was DISCARDED and the texture rendered at the
    wrong scale and offset with nothing reporting a loss. Blender's own 4.1 splash scene
    contains 26 of them. A missing plugin is a visible error; a dropped input is a wrong
    picture, which is worse.

    All four modes are Blender's, verbatim (`kernel/svm/mapping_util.h`):

        POINT    rot * (v * scale) + location
        TEXTURE  safe_divide(rot^T * (v - location), scale)     (the inverse of POINT)
        VECTOR   rot * (v * scale)                              (a direction: no translation)
        NORMAL   normalize(rot * safe_divide(v, scale))

    NORMAL used to be refused here on the grounds that it "involves an inverse-transpose
    whose exact Blender spelling this implementation has not been checked against". Checked:
    it does not. Cycles divides by the scale rather than multiplying, rotates, and
    normalises -- which is the inverse-transpose only because a rotation's inverse is its
    transpose and a diagonal scale's inverse-transpose is its reciprocal. The refusal was
    right in kind (do not guess a normal transform) and wrong in fact, and the fix was to
    read the source rather than to reason about it.

    Attributes
    ----------
    vector : mi.Texture
        The coordinate being transformed.
    location, rotation, scale : mi.Texture
        The transform, each linkable exactly as in Blender. Rotation is Euler XYZ, radians.
    vector_type : str
        'POINT' (default), 'TEXTURE', 'VECTOR' or 'NORMAL'.
    '''

    SUPPORTED = ('POINT', 'TEXTURE', 'VECTOR', 'NORMAL')

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.vector_type = props.get('vector_type', 'POINT')
        if self.vector_type not in Mapping.SUPPORTED:
            raise ValueError(
                f'Mapping: vector_type {self.vector_type} is not supported '
                f'(supported: {", ".join(Mapping.SUPPORTED)})')
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        self.vector   = props.get_unbounded_texture('vector', 0.0)
        self.location = props.get_unbounded_texture('location', 0.0)
        self.rotation = props.get_unbounded_texture('rotation', 0.0)
        self.scale    = props.get_unbounded_texture('scale', 1.0)

    def traverse(self, cb):
        cb.put('vector', self.vector, mi.ParamFlags.Differentiable)
        cb.put('location', self.location, mi.ParamFlags.Differentiable)
        cb.put('rotation', self.rotation, mi.ParamFlags.Differentiable)
        cb.put('scale', self.scale, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_1(self, si, active):
        return self.eval_3(si, active).x

    def eval_3(self, si, active):
        v   = mi.Vector3f(self.vector.eval_3(si, active))
        loc = mi.Vector3f(self.location.eval_3(si, active))
        rot = mi.Vector3f(self.rotation.eval_3(si, active))
        scl = mi.Vector3f(self.scale.eval_3(si, active))
        r0, r1, r2 = euler_xyz_to_mat3(rot)

        if self.vector_type == 'TEXTURE':
            d = v - loc
            # rot^T @ d -- the rows become the columns.
            rotated = mi.Vector3f(dr.dot(mi.Vector3f(r0.x, r1.x, r2.x), d),
                                  dr.dot(mi.Vector3f(r0.y, r1.y, r2.y), d),
                                  dr.dot(mi.Vector3f(r0.z, r1.z, r2.z), d))
            out = mi.Vector3f(dr.select(scl.x != 0.0, rotated.x / scl.x, 0.0),
                              dr.select(scl.y != 0.0, rotated.y / scl.y, 0.0),
                              dr.select(scl.z != 0.0, rotated.z / scl.z, 0.0))
        elif self.vector_type == 'NORMAL':
            # DIVIDES by the scale -- a normal transforms by the inverse transpose, and for a
            # diagonal scale that is the reciprocal, not the scale itself.
            d = mi.Vector3f(dr.select(scl.x != 0.0, v.x / scl.x, 0.0),
                            dr.select(scl.y != 0.0, v.y / scl.y, 0.0),
                            dr.select(scl.z != 0.0, v.z / scl.z, 0.0))
            out = mi.Vector3f(dr.dot(r0, d), dr.dot(r1, d), dr.dot(r2, d))
            n = dr.norm(out)
            out = dr.select(n > 0.0, out / dr.select(n > 0.0, n, 1.0), 0.0)
        else:
            s = v * scl
            out = mi.Vector3f(dr.dot(r0, s), dr.dot(r1, s), dr.dot(r2, s))
            if self.vector_type == 'POINT':
                out = out + loc

        return mi.Color3f(out)

    def mean(self):
        return self.vector.mean()

    def resolution(self):
        return self.vector.resolution()

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return f'Mapping[type={self.vector_type}, vector={self.vector}]'


mi.register_texture('mapping', lambda props: Mapping(props))
