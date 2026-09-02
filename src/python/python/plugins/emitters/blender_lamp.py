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


class BlenderLamp(mi.Emitter):
    '''
    Blender/Cycles LAMP: an area emitter whose radiance is modulated by Cycles' own
    directional attenuation -- the spot cone (mode 'SPOT') or the area-light spread
    (mode 'AREA_SPREAD').

    Cycles models a spot light with a non-zero `shadow_soft_size` as a SPHERE
    (`spot.is_sphere`, from `!(b_light.mode & LA_USE_SOFT_FALLOFF)`,
    intern/cycles/blender/light.cpp) with the cone attenuation applied per sampled point,
    and an area light's `spread` as a multiplier on `eval_fac`
    (`area_light_spread_attenuation`, intern/cycles/kernel/light/area.h). Both are
    functions of the EMITTED DIRECTION, which an emitter -- unlike a radiance texture --
    holds in every code path: `si.wi` on the hit path, `-ds.d` in both direct-sampling
    strategies, and the sampled direction itself in `sample_ray`. This plugin therefore
    OWNS the attenuation, instead of the previous mechanism (an `area` emitter whose
    radiance texture read `si.wi`), which needed a local patch to `area.cpp` to populate
    `si.wi` at all four texture-evaluation sites and broke the moment any of them forgot.

    mode = 'SPOT'
        `spot_light_attenuation` (intern/cycles/kernel/light/spot.h):

            smoothstepf((local_ray.z - cos_half_spot_angle) * spot_smooth)

        with `local_ray` the emitted direction in the lamp frame supplied by `to_object`
        (world -> lamp, +Z the cone axis -- Blender's -Z after the exporter's 180-degree
        X rotation), `spot_smooth = 1 / ((1 - cos_half_spot_angle) * blend)`
        (`SpotLight::copy_to_kernel`, intern/cycles/scene/light.cpp), and `smoothstepf`
        the CLAMPED cubic (intern/cycles/util/math_base.h).

        One documented divergence, unchanged from the texture mechanism: Cycles skips the
        attenuation for a shading point INSIDE the sphere (`d_sq > r_sq` guards it). A
        camera or surface inside the bulb of a spot lamp is the only case that differs.

    mode = 'AREA_SPREAD'
        `area_light_spread_attenuation` (intern/cycles/kernel/light/area.h):

            max((tan_half_spread - tan_a) * normalize_spread, 0)

        with `tan_a` the tangent of the angle between the emitted direction and the flat
        lamp's normal. The `tan_half_spread == 0` arm (fully collimated) is reproduced,
        including its factor of pi. Blender's DEFAULT spread of 180 degrees must NOT
        arrive here: Cycles' constants for it are FLT_MAX and 1/FLT_MAX, whose product is
        a subnormal in single precision, so the exporter omits the attenuation entirely.

    Sampling: both modes use the attached shape's own solid-angle sampling
    (`Shape::sample_direction` / `pdf_direction`), so `sample_direction`,
    `eval_direction`, `pdf_direction` and `eval` see the SAME attenuation by
    construction -- MIS-consistency is structural, not patched in. The emitter is
    one-sided, exactly as the `area`-emitter composition it replaces was.
    '''

    def __init__(self, props):
        mi.Emitter.__init__(self, props)
        if mi.is_polarized:
            raise RuntimeError('blender_lamp: polarized variants are not supported.')
        self.mode = props.get('mode', 'SPOT')
        if self.mode not in MODES:
            raise ValueError(f"blender_lamp: unknown mode {self.mode}; expected one of "
                             f"{', '.join(MODES)}.")
        self.m_radiance = props.get_emissive_texture('radiance', 1.0)

        if self.mode == 'SPOT':
            self.to_object = mi.Transform4f(_rows(props.get('to_object', None)))
            self.cos_half_spot_angle = float(props.get('cos_half_spot_angle'))
            self.spot_smooth = float(props.get('spot_smooth'))
        else:
            self.tan_half_spread = float(props.get('tan_half_spread'))
            self.normalize_spread = float(props.get('normalize_spread'))

        self.m_flags = int(mi.EmitterFlags.Surface)
        if self.m_radiance.is_spatially_varying():
            self.m_flags |= int(mi.EmitterFlags.SpatiallyVarying)

    # ------------------------------------------------------------------ attenuation

    def _attenuation(self, d_world, cos_local):
        """Cycles' multiplier for the world emitted direction `d_world`, whose cosine
        against the emitting surface's shading normal is `cos_local` (> 0 in front)."""
        if self.mode == 'SPOT':
            d = dr.normalize(self.to_object @ mi.Vector3f(d_world))
            t = dr.clip((d.z - self.cos_half_spot_angle) * self.spot_smooth, 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        c = cos_local
        s = dr.sqrt(dr.maximum(1.0 - c * c, 0.0))
        if self.tan_half_spread == 0.0:
            # Fully collimated: everything off-axis is dark, and the on-axis value
            # carries the factor of pi from integrating radiance over the hemisphere.
            return dr.select(s > 1e-5 * dr.maximum(c, 1e-30), 0.0, dr.pi)
        tan_a = s / dr.maximum(c, 1e-30)
        return dr.maximum((self.tan_half_spread - tan_a) * self.normalize_spread, 0.0)

    # ------------------------------------------------------------------ radiance paths

    def eval(self, si, active=True):
        visible = mi.Frame3f.cos_theta(si.wi) > 0.0
        atten = self._attenuation(si.to_world(si.wi), mi.Frame3f.cos_theta(si.wi))
        spec = self.m_radiance.eval(si, active) * atten
        return dr.select(mi.Mask(active) & visible, spec, 0.0)

    def sample_direction(self, it, sample, active=True):
        shape = self.get_shape()
        ds = shape.sample_direction(it, sample, active)
        # `Shape::sample_direction` converts to solid angle through `abs_dot`, so the
        # density is side-agnostic; the front-side test lives here.
        dp = dr.dot(ds.d, ds.n)
        active = mi.Mask(active) & (dp < 0.0) & (ds.pdf != 0.0)

        si = mi.SurfaceInteraction3f(ds, it.wavelengths)
        # `ds.d` points from the receiver TO the emitter; the emitted direction is its
        # negation. Supplied to the radiance texture too, so an ordinary texture reads
        # `si.uv` as always and a direction-aware one is not lied to.
        si.wi = si.to_local(-ds.d)
        spec = self.m_radiance.eval(si, active) * self._attenuation(-ds.d, -dp) / ds.pdf

        ds.emitter = mi.EmitterPtr(self)
        return ds, dr.select(active, spec, 0.0)

    def pdf_direction(self, it, ds, active=True):
        dp = dr.dot(ds.d, ds.n)
        active = mi.Mask(active) & (dp < 0.0)
        return dr.select(active, self.get_shape().pdf_direction(it, ds, active), 0.0)

    def eval_direction(self, it, ds, active=True):
        dp = dr.dot(ds.d, ds.n)
        active = mi.Mask(active) & (dp < 0.0)
        si = mi.SurfaceInteraction3f(ds, it.wavelengths)
        si.wi = si.to_local(-ds.d)
        spec = self.m_radiance.eval(si, active) * self._attenuation(-ds.d, -dp)
        return dr.select(active, spec, 0.0)

    # ------------------------------------------------------------------ endpoint rest

    def sample_position(self, time, sample, active=True):
        ps = self.get_shape().sample_position(time, sample, active)
        weight = dr.select(mi.Mask(active) & (ps.pdf > 0.0), dr.rcp(ps.pdf), 0.0)
        return ps, weight

    def pdf_position(self, ps, active=True):
        return self.get_shape().pdf_position(ps, active)

    def sample_wavelengths(self, si, sample, active=True):
        if mi.is_spectral:
            return self.m_radiance.sample_spectrum(si, mi.sample_shifted(sample), active)
        return self.m_radiance.sample_spectrum(si, dr.zeros(mi.Color0f), active)

    def pdf_wavelengths(self, wavelengths, active=True):
        # Matches stock `area`: the base class raises, and nothing on the path/NEE code
        # path calls it for surface emitters.
        raise NotImplementedError('blender_lamp::pdf_wavelengths')

    def sample_ray(self, time, wavelength_sample, sample2, sample3, active=True):
        # Position, then a cosine-weighted direction; the cosine cancels against the
        # hemisphere pdf, leaving the pi below (as in `area::sample_ray`). The
        # attenuation multiplies the carried radiance.
        ps, pos_weight = self.sample_position(time, sample2, active)
        local = mi.warp.square_to_cosine_hemisphere(sample3)

        si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.SurfaceInteraction3f).wavelengths)
        si.wi = local
        wavelength, wav_weight = self.sample_wavelengths(si, wavelength_sample, active)
        si.time = time
        si.wavelengths = wavelength

        d_world = si.to_world(local)
        weight = pos_weight * wav_weight * dr.pi * self._attenuation(d_world, local.z)
        return si.spawn_ray(d_world), weight

    def bbox(self):
        return self.get_shape().bbox()

    def traverse(self, cb):
        cb.put('radiance', self.m_radiance, +mi.ParamFlags.Differentiable)

    def to_string(self):
        return f'BlenderLamp[{self.mode}, radiance={self.m_radiance}]'


mi.register_emitter('blender_lamp', lambda props: BlenderLamp(props))
