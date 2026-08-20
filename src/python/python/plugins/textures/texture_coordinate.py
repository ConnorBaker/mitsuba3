from __future__ import annotations # Delayed parsing of type annotations
from typing import Tuple

import drjit as dr
import mitsuba as mi

CHANNELS = ('UV', 'Object', 'Generated', 'Normal')


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
    Generated   (to_object * P - gen_min) / gen_size

    Camera, Window and Reflection are absent: each depends on the sensor rather than the
    surface, and a coordinate that silently ignores the camera is a wrong picture.

    GENERATED IS EXACT ONLY FOR UNDEFORMED GEOMETRY, and that is enforced on the exporter
    side rather than here. Blender's Generated is a per-vertex attribute captured from the
    ORIGINAL mesh coordinates and interpolated across the face; the affine map below
    reproduces it exactly when the evaluated mesh has the original's vertex positions, and
    diverges as soon as a modifier moves them.
    '''

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.channel = props.get('channel', 'UV')
        if self.channel not in CHANNELS:
            raise ValueError(f"TextureCoordinate: Invalid channel {self.channel}; "
                             f"supported: {', '.join(CHANNELS)}")
        self.to_object = mi.Transform4f(props.get('to_object', mi.ScalarTransform4f()))
        self.to_world = self.to_object.inverse()
        self.gen_min = mi.Vector3f(props.get('generated_min', mi.ScalarPoint3f(0.0)))
        self.gen_size = mi.Vector3f(props.get('generated_size', mi.ScalarPoint3f(1.0)))

    def _eval_channel(self, si, active):
        if self.channel == 'UV':
            return mi.Vector3f(si.uv.x, 1.0-si.uv.y, 0.0) # Follow Blender's convention
        if self.channel == 'Normal':
            n = mi.Vector3f(si.n)
            m = self.to_world.matrix
            # to_world^T * n -- a normal transforms by the inverse transpose, and going
            # world -> object that is the TRANSPOSE of the object -> world matrix.
            out = mi.Vector3f(m[0][0] * n.x + m[1][0] * n.y + m[2][0] * n.z,
                              m[0][1] * n.x + m[1][1] * n.y + m[2][1] * n.z,
                              m[0][2] * n.x + m[1][2] * n.y + m[2][2] * n.z)
            norm = dr.norm(out)
            return dr.select(norm > 0.0, out / dr.select(norm > 0.0, norm, 1.0), 0.0)
        p = mi.Vector3f(self.to_object @ mi.Point3f(si.p))
        if self.channel == 'Object':
            return p
        return mi.Vector3f(
            dr.select(self.gen_size.x != 0.0, (p.x - self.gen_min.x) / self.gen_size.x, 0.0),
            dr.select(self.gen_size.y != 0.0, (p.y - self.gen_min.y) / self.gen_size.y, 0.0),
            dr.select(self.gen_size.z != 0.0, (p.z - self.gen_min.z) / self.gen_size.z, 0.0))

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
