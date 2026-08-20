from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi


class Gradient(mi.Texture):
    '''
    Gradient Texture Blender shader node.

    Seven closed-form functions of the input coordinate. Every one is checked against Cycles
    by rendering the same node graph through both engines rather than by reading it off a
    source -- see the node-parity harness; a formula that is merely plausible renders without
    complaining, which is the failure mode this plugin set keeps producing.

    Attributes
    ----------
    vector : mi.Texture
        The coordinate. Blender defaults it to Generated coordinates; the exporter supplies
        whatever the node's Vector input is actually wired to.
    gradient_type : str
        LINEAR, QUADRATIC, EASING, DIAGONAL, RADIAL, SPHERICAL or QUADRATIC_SPHERE.
    '''

    TYPES = ('LINEAR', 'QUADRATIC', 'EASING', 'DIAGONAL', 'RADIAL',
             'SPHERICAL', 'QUADRATIC_SPHERE')

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.gradient_type = props.get('gradient_type', 'LINEAR')
        if self.gradient_type not in Gradient.TYPES:
            raise ValueError(
                f'Gradient: gradient_type {self.gradient_type} is not supported '
                f'(supported: {", ".join(Gradient.TYPES)})')
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        self.vector = props.get_unbounded_texture('vector', 0.0)

    def traverse(self, cb):
        cb.put('vector', self.vector, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def _fac(self, si, active):
        p = mi.Vector3f(self.vector.eval_3(si, active))
        t = self.gradient_type
        if t == 'LINEAR':
            f = p.x
        elif t == 'QUADRATIC':
            r = dr.maximum(p.x, 0.0)
            f = r * r
        elif t == 'EASING':
            r = dr.clip(p.x, 0.0, 1.0)
            f = 3.0 * r * r - 2.0 * r * r * r
        elif t == 'DIAGONAL':
            f = (p.x + p.y) * 0.5
        elif t == 'RADIAL':
            f = dr.atan2(p.y, p.x) / (2.0 * dr.pi) + 0.5
        else:
            # The 1e-6 bias is Blender's: for a unit-length coordinate it makes the result
            # exactly zero rather than a float-precision speck.
            r = dr.maximum(1.0 - dr.norm(p) + 1e-6, 0.0)
            f = r * r if t == 'QUADRATIC_SPHERE' else r
        return dr.clip(f, 0.0, 1.0)

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self._fac(si, active))

    def eval_1(self, si, active):
        return mi.Float(self._fac(si, active))

    def eval_3(self, si, active):
        return mi.Color3f(self._fac(si, active))

    def mean(self):
        return mi.Float(0.5)

    def resolution(self):
        return self.vector.resolution()

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return f'Gradient[type={self.gradient_type}, vector={self.vector}]'


mi.register_texture('gradient', lambda props: Gradient(props))
