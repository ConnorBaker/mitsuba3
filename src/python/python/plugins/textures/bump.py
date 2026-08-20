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

    THE DIFFERENCING SCALE IS THE WHOLE BALLGAME, AND A UV GRADIENT IS NOT ENOUGH.
    Cycles computes `surfgrad = (h_x - h_c)*Rx + (h_y - h_c)*Ry` with `Rx = dP.dy x N`,
    `Ry = N x dP.dx`, `det = dP.dx . Rx`, then
    `N' = normalize(fw*|det|*N - dist*sign(det)*surfgrad)` (`kernel/svm/displace.h`), where
    `h_x` and `h_y` are the height re-evaluated at the shading point offset by the
    RAY-DIFFERENTIAL footprint `dP.dx*fw` and `dP.dy*fw`.

    This docstring used to argue that substituting (u, v) for (x, y) is equivalent, because
    `surfgrad/det` is the intrinsic tangential gradient and therefore basis-independent.
    That is true of the EXACT gradient and false of the DISCRETE one, which is what both
    renderers actually compute: Cycles differences over a screen-space PIXEL FOOTPRINT and
    the old code differenced over one TEXEL. Measured on `Fabric Sofa` from the Blender
    splash scene (sphere-centre mean, Mitsuba/Cycles):

        render  40px  1.0780        render 160px  1.5192
        render  80px  1.3550        render 320px  1.6590

    Cycles moves 35% across that ladder -- its bump is a screen-space effect by design --
    while the old implementation was flat to four significant digits, because a texel is the
    same size however many pixels cover it. Sweeping the old `uv_step` at fixed resolution
    gave 0.7391 / 1.1479 / 1.3550 / 2.0596 / 2.5925 for 1/2048 ... 1/16, so the scale was
    exactly the free parameter and no constant value is right for more than one scene.

    So the footprint is now taken from `si.duv_dx` / `si.duv_dy`, which is the same quantity
    Cycles uses, and the arithmetic below is Cycles' line for line -- FORWARD differences
    (`h_x - h_c`, not a central difference), `Rx`/`Ry`/`det` built from the screen-space
    surface derivatives rather than from `dp_du`/`dp_dv`, `max(strength, 0)`, and the final
    `normalize(mix(N, N', strength))`.

    THE FALLBACK, AND WHY IT IS NOT SILENT. `duv_dx` is only populated where a ray carries
    differentials -- the camera hit. Mitsuba does not propagate them past the first bounce
    (Cycles carries an approximate widened differential), so at deeper vertices this falls
    back to an axis-aligned texel-sized footprint, i.e. the old behaviour. Bump seen
    directly is now correct; bump seen in a mirror or after a rough bounce is still
    differenced at texel scale. That is a known residual, recorded here and in the
    integrator, not a silent substitution.

    WHY NOT `eval_1_grad`. It is exact but implemented by `bitmap` and `uniform` only --
    every Python-side Blender node plugin inherits the pure-virtual stub, so a Bump fed by a
    Color Ramp, a Math chain or a Mix (which is most of them) would fault at render time.

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
    filter_width : mi.Texture
        Cycles' Bump node `Filter Width` input, which scales the footprint the height field
        is differenced over. Blender's default is 0.1 and so is this one.
    uv_step : float
        Fallback differencing step in UV, used only where the shading point carries no ray
        differentials. Default 0, meaning "derive it from the height texture's resolution".
    '''

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        # HSR: `get_texture` builds an SRGBReflectanceSpectrum from an `rgb` constant, which
        # REJECTS any component outside [0, 1]. These are coordinates and arithmetic operands, not
        # reflectances -- a Mapping location of -0.27, a Vector Math operand of 2.0 and a Math
        # operand of -1.0 are all ordinary -- so they take the UNBOUNDED form. This was not a
        # theoretical concern: it was found by rendering a Mapping node with a negative Location.
        self.height   = props.get_unbounded_texture('height', 0.0)
        self.strength = props.get_unbounded_texture('strength', 1.0)
        self.distance = props.get_unbounded_texture('distance', 1.0)
        # Blender's Bump node default; `SOCKET_IN_FLOAT(filter_width, "Filter Width", 0.1f)`
        # in `intern/cycles/scene/shader_nodes.cpp`.
        self.filter_width = props.get_unbounded_texture('filter_width', 0.1)
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
        cb.put('filter_width', self.filter_width, mi.ParamFlags.Differentiable)
        if self.has_normal:
            cb.put('normal', self.normal, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def _height_at(self, si, du, dv, active):
        s = mi.SurfaceInteraction3f(si)
        s.uv = mi.Point2f(si.uv.x + du, si.uv.y + dv)
        return self.height.eval_1(s, active)

    def _footprint(self, si):
        """The UV offsets to difference the height field over: (duv_dx, duv_dy).

        Where the shading point carries ray differentials this is the screen-space pixel
        footprint, which is what Cycles uses. Where it does not -- every vertex past the
        camera hit, since Mitsuba drops differentials there -- it degenerates to an
        axis-aligned texel-sized step, chosen per lane rather than globally so a scene with
        both kinds of vertex gets each treated correctly.
        """
        dx, dy = mi.Vector2f(si.duv_dx), mi.Vector2f(si.duv_dy)
        have = (dr.abs(dx.x) + dr.abs(dx.y) + dr.abs(dy.x) + dr.abs(dy.y)) > 0.0
        fx = mi.Vector2f(dr.select(have, dx.x, self.step_u), dr.select(have, dx.y, 0.0))
        fy = mi.Vector2f(dr.select(have, dy.x, 0.0), dr.select(have, dy.y, self.step_v))
        return fx, fy

    def _perturbed_normal(self, si, active):
        if self.has_normal:
            n = dr.normalize(si.to_world(
                mi.Normal3f(2.0 * self.normal.eval_3(si, active) - 1.0)))
        else:
            n = mi.Normal3f(si.sh_frame.n)

        fw = self.filter_width.eval_1(si, active)
        duv_dx, duv_dy = self._footprint(si)
        duv_dx = duv_dx * fw
        duv_dy = duv_dy * fw

        # Screen-space surface derivatives, exactly Cycles' `dP.dx` / `dP.dy`, expressed in
        # the basis this renderer does carry: dP/dx = dp_du * du/dx + dp_dv * dv/dx.
        dp_du = mi.Vector3f(si.dp_du)
        dp_dv = mi.Vector3f(si.dp_dv)
        dp_dx = dp_du * duv_dx.x + dp_dv * duv_dx.y
        dp_dy = dp_du * duv_dy.x + dp_dv * duv_dy.y

        # FORWARD differences at the offset points -- `h_x - h_c`, not a central difference.
        h_c = self._height_at(si, 0.0, 0.0, active)
        h_x = self._height_at(si, duv_dx.x, duv_dx.y, active)
        h_y = self._height_at(si, duv_dy.x, duv_dy.y, active)

        Rx = dr.cross(dp_dy, n)
        Ry = dr.cross(n, dp_dx)
        det = dr.dot(dp_dx, Rx)
        surfgrad = (h_x - h_c) * Rx + (h_y - h_c) * Ry

        dist = self.distance.eval_1(si, active)
        if self.invert:
            dist = -dist
        strength = dr.maximum(self.strength.eval_1(si, active), 0.0)

        perturbed = dr.normalize(fw * dr.abs(det) * n - dist * dr.sign(det) * surfgrad)
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
                f'distance={self.distance}, filter_width={self.filter_width}, '
                f'invert={self.invert}]')


mi.register_texture('bump', lambda props: Bump(props))
