# Blender's Blackbody node -- `kernel/svm/blackbody.h`.
#
# The colour is a PIECEWISE FIT of the Planckian locus in Rec.709 with seven temperature
# knots, not a Planck evaluation; reproducing the physics instead of the fit would give a
# defensible answer that does not match the render.

import drjit as dr
import mitsuba as mi

from .blender_color import blackbody_rec709


class Blackbody(mi.Texture):
    """Blackbody."""

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.temperature = props.get_unbounded_texture('temperature', 1500.0)

    def _color(self, si, active):
        # Cycles applies `rec709_to_rgb` afterwards. In an RGB-variant Mitsuba the rendering
        # primaries ARE Rec.709 -- Blender's own default scene-linear space -- so that step
        # is the identity here. It stops being the identity in a spectral variant, which is
        # why this is stated rather than assumed.
        return blackbody_rec709(self.temperature.eval_1(si, active))

    def eval(self, si, active=True):
        return mi.UnpolarizedSpectrum(self._color(si, active))

    def eval_1(self, si, active=True):
        c = self._color(si, active)
        return (c.x + c.y + c.z) / 3.0

    def eval_3(self, si, active=True):
        return self._color(si, active)

    def mean(self):
        return 1.0

    def traverse(self, cb):
        cb.put('temperature', self.temperature, mi.ParamFlags.Differentiable)

    def to_string(self):
        return 'Blackbody[temperature=%s]' % self.temperature


mi.register_texture('blackbody', lambda props: Blackbody(props))
