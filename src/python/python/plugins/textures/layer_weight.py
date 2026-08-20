from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

# Blender's Layer Weight node, transcribed from Cycles' `svm_node_layer_weight`
# (intern/cycles/kernel/svm/fresnel.h) and `fresnel_dielectric_cos`
# (intern/cycles/kernel/closure/bsdf_util.h):
#
#   FRESNEL   eta = max(1 - blend, 1e-5); eta = backfacing ? eta : 1/eta
#             f   = fresnel_dielectric_cos(dot(wi, N), eta)
#   FACING    f   = |dot(wi, N)|
#             if blend != 0.5: blend = clamp(blend, 0, 1 - 1e-5)
#                              blend = blend < 0.5 ? 2*blend : 0.5/(1 - blend)
#                              f = pow(f, blend)
#             f   = 1 - f
#
# `dot(wi, N)` is exactly `si.wi.z` in Mitsuba's shading frame, so no world-space normal is
# needed -- which is also why a LINKED Normal input is refused on the exporter side rather
# than approximated here: it would arrive in Blender's world space, and using the shading
# frame for it would silently be a different picture.
#
# The `blend != 0.5` guard is Cycles' own and is kept as an EXACT float comparison rather
# than a tolerance: at exactly 0.5 the reshaping is skipped, and 0.5 is the node's default,
# so the common case must take the untouched branch.

WEIGHT_TYPES = ('FRESNEL', 'FACING')


def _fresnel_dielectric_cos(cosi, eta):
    c = dr.abs(cosi)
    g = eta * eta - 1.0 + c * c
    ok = g > 0.0
    gs = dr.sqrt(dr.maximum(g, 0.0))
    A = (gs - c) / (gs + c)
    B = (c * (gs + c) - 1.0) / (c * (gs - c) + 1.0)
    return dr.select(ok, 0.5 * A * A * (1.0 + B * B), 1.0)   # g <= 0 is TIR


class LayerWeight(mi.Texture):
    '''Blender's Layer Weight shader node.'''

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.weight_type = str(props.get('weight_type', 'FRESNEL'))
        if self.weight_type not in WEIGHT_TYPES:
            raise RuntimeError("layer_weight: output '%s' is not supported; %s are."
                               % (self.weight_type, ', '.join(WEIGHT_TYPES)))
        self.blend = props.get_unbounded_texture('blend', 0.5)

    def _eval_f(self, si, active):
        blend = self.blend.eval_1(si, active)
        cosi = si.wi.z
        if self.weight_type == 'FRESNEL':
            eta = dr.maximum(1.0 - blend, 1e-5)
            eta = dr.select(cosi < 0.0, eta, dr.rcp(eta))
            return _fresnel_dielectric_cos(cosi, eta)
        f = dr.abs(cosi)
        b = dr.clip(blend, 0.0, 1.0 - 1e-5)
        b = dr.select(b < 0.5, 2.0 * b, 0.5 / dr.maximum(1.0 - b, 1e-5))
        return 1.0 - dr.select(blend == 0.5, f, dr.power(f, b))

    def eval(self, si, active=True):
        return mi.UnpolarizedSpectrum(self._eval_f(si, active))

    def eval_1(self, si, active=True):
        return self._eval_f(si, active)

    def eval_3(self, si, active=True):
        return mi.Color3f(self._eval_f(si, active))

    def mean(self):
        return mi.Float(0.5)

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return f'LayerWeight[{self.weight_type}]'


mi.register_texture('layer_weight', lambda props: LayerWeight(props))
