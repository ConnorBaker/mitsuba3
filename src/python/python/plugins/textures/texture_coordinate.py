from __future__ import annotations # Delayed parsing of type annotations
from typing import Tuple

import drjit as dr
import mitsuba as mi

CHANNELS = ('UV', 'Object', 'Generated', 'Normal')
# Generated is deliberately NOT here: it is a mesh attribute now, so it needs no object
# frame -- which is exactly what lets it survive instancing and particle systems.
OBJECT_SPACE = ('Object', 'Normal')


def _rows(value):
    """A 4x4 as a list of Python float ROWS, from a transform, a matrix or a nested list."""
    if value is None:
        return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    if isinstance(value, (list, tuple)):
        return [[float(value[i][j]) for j in range(4)] for i in range(4)]
    m = getattr(value, 'matrix', value)
    try:
        return [[float(m[i][j]) for j in range(4)] for i in range(4)]
    except Exception:
        return [[float(m[j][i]) for j in range(4)] for i in range(4)]


def _invert(rows):
    """Gauss-Jordan on a 4x4 of Python floats. Small, exact enough, and no dependency on a
    transform class whose inverse would have to round-trip through a type we cannot name."""
    a = [list(rows[i]) + [1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-20:
            raise ValueError('TextureCoordinate: to_object is singular')
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [v / d for v in a[col]]
        for r in range(4):
            if r == col:
                continue
            f = a[r][col]
            if f:
                a[r] = [v - f * w for v, w in zip(a[r], a[col])]
    return [row[4:] for row in a]


class TextureCoordinate(mi.Texture):
    '''
    Texture coordinates Blender shader node texture.

    Four of Blender's seven outputs. UV is the mesh parameterisation; the other three are
    OBJECT-SPACE, which is why they carry a `to_object` transform: a Mitsuba
    `SurfaceInteraction` knows only world space, so the object frame has to be supplied by
    whoever exported the object. That also makes a material using them OBJECT-DEPENDENT --
    two objects sharing one Blender material need two Mitsuba materials, which the exporter
    handles by keying the material id on the object.

    Object      to_object * P                       (the object-space position)
    Normal      normalize(to_world^T * N)           (inverse-transpose, i.e. a NORMAL)
    Generated   the `vertex_generated` MESH ATTRIBUTE, barycentrically interpolated

    Camera, Window and Reflection are absent: each depends on the sensor rather than the
    surface, and a coordinate that silently ignores the camera is a wrong picture.

    GENERATED IS NOT A FUNCTION OF THE SURFACE POSITION and is not computed here. Cycles'
    `attr_create_generated` (intern/cycles/blender/mesh.cpp) reads the CD_ORCO layer and
    derives Generated from the UNDEFORMED vertex coordinates, falling back to the evaluated
    positions only when that layer is absent. An affine map of the positions this plugin
    can see therefore reproduces it only for geometry no modifier moved -- which is how
    this used to be written, with the divergence pushed onto the exporter as a refusal.
    It is now carried as a per-vertex attribute instead, which is exact under deforming and
    generative modifiers, and under instancing and particle systems, where there is no
    single object frame to map through.
    '''

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.channel = props.get('channel', 'UV')
        if self.channel not in CHANNELS:
            raise ValueError(f"TextureCoordinate: Invalid channel {self.channel}; "
                             f"supported: {', '.join(CHANNELS)}")
        # Read as PLAIN FLOATS and rebuild. `props.get` hands back whatever precision and
        # transform class the caller happened to construct (a ScalarAffineTransform4d, in
        # practice), and `Transform4f` accepts none of them -- so the conversion is done
        # here, once, rather than depending on which overload the caller's type matches.
        raw = props.get('to_object', None) if self.channel in OBJECT_SPACE else None
        rows = _rows(raw)
        self.to_object = mi.Transform4f(rows)
        # The object -> world matrix, kept as Python floats: a NORMAL transforms by the
        # inverse transpose, which going world -> object is exactly this matrix transposed,
        # and doing the transpose on scalars avoids depending on whether drjit's Matrix4f
        # indexes rows or columns.
        self.o2w = _rows(_invert(rows))
        # The mesh attribute Generated is read from. The exporter bakes Blender's texture
        # space into it (`generated = (undeformed_co - (loc - size)) / (2 * size)`), because
        # that is where the texspace values live; nothing is recomputed here.
        self.generated_attribute = str(props.get('generated_attribute', 'vertex_generated'))

    def _eval_channel(self, si, active):
        if self.channel == 'UV':
            return mi.Vector3f(si.uv.x, 1.0-si.uv.y, 0.0) # Follow Blender's convention
        if self.channel == 'Normal':
            n = mi.Vector3f(si.n)
            m = self.o2w
            out = mi.Vector3f(m[0][0] * n.x + m[1][0] * n.y + m[2][0] * n.z,
                              m[0][1] * n.x + m[1][1] * n.y + m[2][1] * n.z,
                              m[0][2] * n.x + m[1][2] * n.y + m[2][2] * n.z)
            norm = dr.norm(out)
            return dr.select(norm > 0.0, out / dr.select(norm > 0.0, norm, 1.0), 0.0)
        if self.channel == 'Generated':
            # Loud on a mesh that does not carry it: `Mesh::eval_attribute_3` throws for an
            # unknown name, which is the right outcome -- a Generated coordinate silently
            # reading zero is a wrong picture, and the exporter is what guarantees the
            # attribute is present.
            return mi.Vector3f(si.shape.eval_attribute_3(
                self.generated_attribute, si, active))
        return mi.Vector3f(self.to_object @ mi.Point3f(si.p))

    def eval(self, si, active):
        # HSR: used to raise. `eval` is the generic entry point -- an `area` emitter calls it
        # to get radiance, and every wrapper plugin (`math`, `mix_color`, ...) forwards its
        # own `eval` to its inputs' -- so a coordinate anywhere under one of those faulted
        # mid-render even though `eval_3` right beside it would have answered.
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_1(self, si, active):
        # HSR: used to raise. Blender's implicit vector -> float conversion is the MEAN of
        # the three components, and a coordinate wired into a float socket (a Color Ramp's
        # Fac, a Math operand) is an ordinary thing to do, so answer with that rather than
        # faulting per-sample.
        v = self.eval_3(si, active)
        return (v.x + v.y + v.z) * (1.0 / 3.0)

    def eval_3(self, si, active):
        return mi.Color3f(self._eval_channel(si, active))

    def mean(self):
        return mi.Float(0.5)

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return f'Texture Coordinate[{self.channel}]'

mi.register_texture('texture_coordinate', lambda props: TextureCoordinate(props))
