from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi


class Bump(mi.Texture):
    '''
    Bump Blender shader node: a height field perturbing the shading normal.

    OUTPUT ENCODING. This is a TEXTURE, not a BSDF adapter, because that is what the
    consumer needs: `blender_principled` takes its `normalmap` as a [0, 1]-encoded normal in
    the LOCAL shading frame and decodes it with `2 x - 1` (`compute_normalmap_frame`). So
    this plugin returns the perturbed normal in exactly that encoding and drops straight into
    the same socket a Normal Map node would feed -- including when a Normal Map feeds THIS
    node's `normal` input, which is how the two are chained in Blender.

    WHY A UV GRADIENT REPRODUCES CYCLES' SCREEN-SPACE ONE. Cycles computes
    `surfgrad = dHdx*Rx + dHdy*Ry` with `Rx = dPdy x N`, `Ry = N x dPdx`, `det = dPdx.Rx`
    from RAY DIFFERENTIALS, then
    `N' = normalize(|det|*N - dist*sign(det)*surfgrad)`. The quantity `surfgrad/det` is the
    INTRINSIC tangential gradient of the height field: it is the same vector for any basis
    spanning the tangent plane, so substituting (u, v) for (x, y) -- `dp_du`, `dp_dv` and the
    UV derivatives of H -- yields the identical perturbed normal without needing differentials
    that an offline path tracer does not carry per-shading-point. That is also why the whole
    expression is invariant to a flipped v axis: `surfgrad` and `det` both change sign.
    The arithmetic below is Cycles' line for line, including `max(strength, 0)` and the final
    `normalize(mix(N, N', strength))`.

    THE GRADIENT IS A CENTRAL DIFFERENCE, DELIBERATELY. `Texture::eval_1_grad` is exact but
    is implemented by `bitmap` and `uniform` only -- every Python-side Blender node plugin
    inherits the pure-virtual stub, so a Bump fed by a Color Ramp, a Math chain or a Mix
    (which is most of them) would fault at render time. A central difference works on all of
    them and is the same quantity. Its step is one texel of the height texture's own
    resolution where that is known, so a bitmap is differenced at the scale it actually
    carries detail at, and `uv_step` overrides it.

    Attributes
    ----------
    height : mi.Texture
        The scalar height field. A constant makes this node a no-op (zero gradient).
    strength : mi.Texture
        Blend between the unperturbed and perturbed normal. Clamped at 0 below. Default 1.
    distance : mi.Texture
        Scales the perturbation. Default 1.
    invert : bool
        Negates `distance`, i.e. turns bumps into dents. Default False.
    normal : mi.Texture
        Optional [0, 1]-encoded local-frame normal to perturb INSTEAD of the shading normal
        -- the Bump node's own `Normal` input, which is how a Normal Map chains into a Bump.
    uv_step : float
        Central-difference step in UV. Default 0, meaning "derive it from the height
        texture's resolution".
    '''

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.height   = props.get_texture('height', 0.0)
        self.strength = props.get_texture('strength', 1.0)
        self.distance = props.get_texture('distance', 1.0)
        self.invert   = props.get('invert', False)
        self.has_normal = 'normal' in props.keys()
        self.normal = props.get_texture('normal', 0.0) if self.has_normal else None

        step = float(props.get('uv_step', 0.0))
        if step <= 0.0:
            try:
                res = self.height.resolution()
                res_u, res_v = int(res.x), int(res.y)
            except Exception:
                res_u = res_v = 1
            # A procedural height field reports 1x1; differencing it over a whole UV unit
            # would return the field's endpoints rather than its slope, so floor the
            # resolution at a value fine enough to resolve a smooth signal.
            self.step_u = 1.0 / max(res_u, 256)
            self.step_v = 1.0 / max(res_v, 256)
        else:
            self.step_u = self.step_v = step

    def traverse(self, cb):
        cb.put('height', self.height, mi.ParamFlags.Differentiable)
        cb.put('strength', self.strength, mi.ParamFlags.Differentiable)
        cb.put('distance', self.distance, mi.ParamFlags.Differentiable)
        if self.has_normal:
            cb.put('normal', self.normal, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def _height_at(self, si, du, dv, active):
        s = mi.SurfaceInteraction3f(si)
        s.uv = mi.Point2f(si.uv.x + du, si.uv.y + dv)
        return self.height.eval_1(s, active)

    def _perturbed_normal(self, si, active):
        hu = self.step_u
        hv = self.step_v
        dHdu = (self._height_at(si, hu, 0.0, active)
                - self._height_at(si, -hu, 0.0, active)) / (2.0 * hu)
        dHdv = (self._height_at(si, 0.0, hv, active)
                - self._height_at(si, 0.0, -hv, active)) / (2.0 * hv)

        if self.has_normal:
            n = dr.normalize(si.to_world(
                mi.Normal3f(2.0 * self.normal.eval_3(si, active) - 1.0)))
        else:
            n = mi.Normal3f(si.sh_frame.n)

        dp_du = mi.Vector3f(si.dp_du)
        dp_dv = mi.Vector3f(si.dp_dv)
        Rx = dr.cross(dp_dv, n)
        Ry = dr.cross(n, dp_du)
        det = dr.dot(dp_du, Rx)
        surfgrad = dHdu * Rx + dHdv * Ry

        dist = self.distance.eval_1(si, active)
        if self.invert:
            dist = -dist
        strength = dr.maximum(self.strength.eval_1(si, active), 0.0)

        perturbed = dr.normalize(dr.abs(det) * n - dist * dr.sign(det) * surfgrad)
        # A degenerate tangent frame (det == 0) makes `perturbed` NaN; fall back to the
        # unperturbed normal there rather than propagating NaN into the shading frame, where
        # it turns into black pixels that look like a material bug rather than a bad UV.
        perturbed = dr.select(dr.isnan(perturbed.x) | (det == 0.0), n, perturbed)
        return dr.normalize(strength * perturbed + (1.0 - strength) * n)

    def eval(self, si, active):
        return mi.UnpolarizedSpectrum(self.eval_3(si, active))

    def eval_1(self, si, active):
        raise ValueError('Bump: eval_1 not supported!')

    def eval_3(self, si, active):
        n = self._perturbed_normal(si, active)
        return mi.Color3f((si.to_local(n) + 1.0) * 0.5)

    def mean(self):
        return mi.Float(0.5)

    def is_spatially_varying(self):
        return True

    def to_string(self):
        return (f'Bump[height={self.height}, strength={self.strength}, '
                f'distance={self.distance}, invert={self.invert}]')


mi.register_texture('bump', lambda props: Bump(props))
