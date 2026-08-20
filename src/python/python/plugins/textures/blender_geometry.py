from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

# Blender's Geometry input node. Cycles' `svm_node_geometry_eval`
# (intern/cycles/kernel/svm/geometry.h) is a straight switch over the shading point:
#
#   Position       sd->P      True Normal   sd->Ng      Incoming    sd->wi
#   Normal         sd->N      Parametric    (1-u-v, u, v)
#
# All of those are world-space quantities in BLENDER's world, which is not necessarily
# Mitsuba's: the exporter applies an axis conversion to every transform it writes. So the
# inverse of that conversion comes in as `to_blender` and is applied to the vectors that
# are directions or positions. It is the identity in the common case, and passing it
# explicitly is what stops that common case from being an assumption.
#
# Tangent, Pointiness and Random are absent rather than approximated: Tangent needs the
# mesh's dPdu convention, Pointiness needs a curvature pass over the mesh, and Random is a
# per-object identifier Mitsuba does not carry. Each is a different picture if guessed.

CHANNELS = ('Position', 'Normal', 'True Normal', 'Incoming', 'Backfacing')


def _rows(value):
    if value is None:
        return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    if isinstance(value, (list, tuple)):
        return [[float(value[i][j]) for j in range(4)] for i in range(4)]
    m = getattr(value, 'matrix', value)
    try:
        return [[float(m[i][j]) for j in range(4)] for i in range(4)]
    except Exception:
        return [[float(m[j][i]) for j in range(4)] for i in range(4)]


class BlenderGeometry(mi.Texture):
    '''Blender's Geometry shader node.'''

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.channel = str(props.get('channel', 'Position'))
        if self.channel not in CHANNELS:
            raise RuntimeError("blender_geometry: output '%s' is not supported; %s are."
                               % (self.channel, ', '.join(CHANNELS)))
        self.to_blender = mi.Transform4f(_rows(props.get('to_blender', None)))

    def _eval_channel(self, si, active):
        if self.channel == 'Position':
            return mi.Vector3f(self.to_blender @ mi.Point3f(si.p))
        if self.channel == 'Normal':
            return mi.Vector3f(self.to_blender @ mi.Vector3f(si.sh_frame.n))
        if self.channel == 'True Normal':
            return mi.Vector3f(self.to_blender @ mi.Vector3f(si.n))
        if self.channel == 'Incoming':
            return mi.Vector3f(self.to_blender @ si.to_world(si.wi))
        # Backfacing: Cycles reports the shading point's SD_BACKFACING flag, which is the
        # viewer being on the far side of the shading normal.
        return mi.Vector3f(dr.select(si.wi.z < 0.0, 1.0, 0.0))

    def eval(self, si, active=True):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_1(self, si, active=True):
        v = self._eval_channel(si, active)
        return (v.x + v.y + v.z) * (1.0 / 3.0)

    def eval_3(self, si, active=True):
        return mi.Color3f(self._eval_channel(si, active))

    def mean(self):
        return mi.Float(0.5)

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return f'BlenderGeometry[{self.channel}]'


mi.register_texture('blender_geometry', lambda props: BlenderGeometry(props))
