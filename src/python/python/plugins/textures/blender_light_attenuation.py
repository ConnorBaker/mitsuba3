from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

MODES = ('SPOT', 'AREA_SPREAD')


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


class BlenderLightAttenuation(mi.Texture):
    '''
    Blender/Cycles LIGHT ATTENUATION, as a radiance texture on an `area` emitter.

    Cycles' spot cone and area-light spread are both functions of the EMITTED DIRECTION, and
    both are applied as a multiplier on the light's `eval_fac`. Mitsuba has no equivalent
    knob on `area`, but an `area` emitter evaluates its `radiance` texture with a
    `SurfaceInteraction` whose `wi` IS the emitted direction (in the local shading frame),
    so the same multiplier can be carried as a texture. That is what this is: it wraps the
    un-attenuated `radiance` and multiplies by Cycles' own factor.

    This is what makes a spot light with a non-zero `shadow_soft_size` expressible at all.
    Cycles models such a light as a SPHERE (`spot.is_sphere`, set from
    `!(b_light.mode & LA_USE_SOFT_FALLOFF)`, intern/cycles/blender/light.cpp) with the cone
    attenuation applied per sampled point, which is exactly `sphere` + `area` + this.

    mode = 'SPOT'
        `spot_light_attenuation` (intern/cycles/kernel/light/spot.h):

            smoothstepf((local_ray.z - cos_half_spot_angle) * spot_smooth)

        where `local_ray` is the direction from the light to the shading point, expressed in
        the lamp frame with z negated so that +z is the cone axis. `to_object` supplies that
        frame (world -> lamp, +Z already the axis, i.e. Blender's -Z after the exporter's
        180-degree X rotation). `spot_smooth = 1 / ((1 - cos_half_spot_angle) * blend)`
        (`SpotLight::copy_to_kernel`, intern/cycles/scene/light.cpp), and `smoothstepf` is
        the CLAMPED cubic (intern/cycles/util/math_base.h).

        One documented divergence: Cycles skips the attenuation for a shading point INSIDE
        the sphere (`d_sq > r_sq` guards it), which cannot be detected from a surface
        interaction on the sphere alone. A camera or surface inside the bulb of a spot light
        is the only case that differs.

    mode = 'AREA_SPREAD'
        `area_light_spread_attenuation` (intern/cycles/kernel/light/area.h):

            max((tan_half_spread - tan_a) * normalize_spread, 0)

        with `tan_a` the tangent of the angle between the emitted direction and the light's
        normal -- which in the local shading frame of a flat light is just
        `sqrt(1 - wi.z^2) / wi.z`. The `tan_half_spread == 0` arm (a fully collimated light)
        is reproduced as well, including its factor of pi. Blender's DEFAULT spread of 180
        degrees is NOT expected to arrive here: Cycles' own constants for it are FLT_MAX and
        1/FLT_MAX, whose product is a subnormal in single precision, so the exporter omits
        the wrapper entirely in that case rather than multiplying by a numerically fragile
        1.0.
    '''

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.mode = props.get('mode', 'SPOT')
        if self.mode not in MODES:
            raise ValueError(f"BlenderLightAttenuation: unknown mode {self.mode}; expected "
                             f"one of {', '.join(MODES)}.")
        self.radiance = props.get_emissive_texture('radiance', 1.0)

        if self.mode == 'SPOT':
            self.to_object = mi.Transform4f(_rows(props.get('to_object', None)))
            self.cos_half_spot_angle = float(props.get('cos_half_spot_angle'))
            self.spot_smooth = float(props.get('spot_smooth'))
        else:
            self.tan_half_spread = float(props.get('tan_half_spread'))
            self.normalize_spread = float(props.get('normalize_spread'))

    def _attenuation(self, si, active):
        if self.mode == 'SPOT':
            # The direction the light emits in, taken out to WORLD and then into the LAMP
            # frame. `si.wi` is in the sphere point's shading frame, which is not the lamp
            # frame -- on a sphere every point has a different one.
            d_world = si.to_world(si.wi)
            d = self.to_object @ mi.Vector3f(d_world)
            d = dr.normalize(d)
            t = dr.clip((d.z - self.cos_half_spot_angle) * self.spot_smooth, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # AREA_SPREAD. A flat light's shading normal IS its `lightNg`, so the angle Cycles
        # measures between the emitted direction and the normal is read straight off wi.z.
        c = si.wi.z
        s = dr.sqrt(dr.maximum(1.0 - c * c, 0.0))
        if self.tan_half_spread == 0.0:
            # Fully collimated: everything off-axis is dark, and the on-axis value carries
            # the factor of pi from integrating radiance over the hemisphere.
            return dr.select(s > 1e-5 * dr.maximum(c, 1e-30), 0.0, dr.pi)
        tan_a = s / dr.maximum(c, 1e-30)
        return dr.maximum((self.tan_half_spread - tan_a) * self.normalize_spread, 0.0)

    def traverse(self, cb):
        cb.put('radiance', self.radiance, +mi.ParamFlags.Differentiable)

    def eval(self, si, active=True):
        return self.radiance.eval(si, active) * self._attenuation(si, active)

    def eval_1(self, si, active=True):
        return self.radiance.eval_1(si, active) * self._attenuation(si, active)

    def eval_3(self, si, active=True):
        return self.radiance.eval_3(si, active) * self._attenuation(si, active)

    def mean(self):
        return self.radiance.mean()

    def is_spatially_varying(self):
        # This answers "does the value depend on si.uv", and the two modes differ.
        #
        # SPOT rides on a SPHERE. Every point of a sphere has a different shading frame, so
        # the same emitted direction gives a different `wi` at each one and the value really
        # does vary across the surface. True.
        #
        # AREA_SPREAD rides on a flat `rectangle` or `disk`, whose shading frame is constant.
        # The value there is a function of the emitted DIRECTION alone and of `si.uv` not at
        # all -- so False, and saying True was wrong on its own terms as well as costly.
        # `area::sample_direction` branches on this: True selects importance sampling of the
        # TEXTURE, which asks for `sample_position` / `pdf_position` this plugin does not
        # implement, so both fall back to the base class and the density that comes back is
        # not the one the sample was drawn from. False selects the shape's own solid-angle
        # sampling, which is the strategy a flat emitter wants anyway.
        return self.mode == 'SPOT'

    def to_string(self):
        return f'BlenderLightAttenuation[{self.mode}, radiance={self.radiance}]'


mi.register_texture('blender_light_attenuation',
                    lambda props: BlenderLightAttenuation(props))
