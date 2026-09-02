from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
from ._base import TextureBase


class Bump(TextureBase):
    '''
    Bump Blender shader node: a height field perturbing the shading normal.

    OUTPUT ENCODING. This is a TEXTURE, not a BSDF adapter, because that is what the
    consumer needs: `blender_principled` takes its `normalmap` as a [0, 1]-encoded normal in
    the LOCAL shading frame and decodes it with `2 x - 1` (`compute_normalmap_frame`). So
    this plugin returns the perturbed normal in exactly that encoding and drops straight into
    the same socket a Normal Map node would feed -- including when a Normal Map feeds THIS
    node's `normal` input, which is how the two are chained in Blender.

    THE DIFFERENCING SCALE IS THE WHOLE BALLGAME, AND IT IS SET OUTSIDE THIS PLUGIN.
    Cycles computes `surfgrad = (h_x - h_c)*Rx + (h_y - h_c)*Ry` with `Rx = dP.dy x N`,
    `Ry = N x dP.dx`, `det = dP.dx . Rx`, then
    `N' = normalize(fw*|det|*N - dist*sign(det)*surfgrad)` (`kernel/svm/displace.h`), where
    `h_x` and `h_y` are the height re-evaluated at the shading point offset by `dP.dx*fw`
    and `dP.dy*fw` -- the offset is applied at the SOURCE nodes, e.g.
    `data.val += data.dx * node.bump_filter_width` in `kernel/svm/tex_coord.h`.

    Three readings of that code were wrong here. All three are kept, because which
    quantity is WRONG is the whole content of this plugin:

    1. This docstring once argued that substituting (u, v) for (x, y) is equivalent, because
       `surfgrad/det` is the intrinsic tangential gradient and therefore basis-independent.
       True of the EXACT gradient, false of the DISCRETE one. Cycles differences over a
       screen-space footprint and the old code differenced over one TEXEL, so on `Fabric
       Sofa` (splash scene, sphere-centre mean, Mitsuba/Cycles) Cycles moved 35% across a
       40/80/160/320 px ladder -- 0.387547 / 0.306080 / 0.273144 / 0.250178 -- while the
       implementation was flat to four digits. A texel is the same size however many pixels
       cover it.

    2. Feeding `si.duv_dx` / `si.duv_dy` straight in is the right MECHANISM at the wrong
       SCALE. It moved the ladder to 0.6415 / 0.6915 / 0.7596 / 0.8355 -- from 35% too
       bright to 30% too dark, and still not tracking. The cause is not in this plugin:
       `SamplingIntegrator::render` shrinks the sensor differential by `rsqrt(spp)`, which
       reconstructs a pixel-wide filter for a LINEAR consumer and does not for a normal
       perturbation. `path.cpp` now divides by `ray_.diff_scale` to hand this plugin the
       one-pixel footprint Cycles carries; see the long comment there for the spp sweep
       that pins it (mi/cy 1.0998 at spp 1, falling to 0.6720 at spp 256, with the Cycles
       arm spread 0.000000).

    3. Cycles' footprint is ISOTROPIC, and this turned out NOT to be the level error --
       recorded because the wrong diagnosis is the useful part. `svm_node_set_bump` reads
       `differential_from_compact(sd->Ng, sd->dP)`, and `sd->dP` is the COMPACTED SCALAR
       `0.5*(|dP.dx| + |dP.dy|)` (`differential_make_compact`); `differential_from_compact`
       rebuilds an isotropic orthonormal pair `(r*ex, r*ey)` from `make_orthonormals(Ng)`.
       So the footprint is a CIRCLE of radius r, not the screen-space parallelogram. The
       argument for why that MUST matter was clean -- `det` is the parallelogram AREA, so at
       grazing incidence the true differential has `det << r^2` while `|surfgrad|` does not
       shrink with it, and the second term of `N'` should run away. Measured, it does not:
       switching to the isotropic reconstruction moved the same ladder to
       0.5829 / 0.6853 / 0.7532 / 0.8333, i.e. nothing at 80/160/320 and slightly WORSE at
       40. It is kept anyway, on the only ground that survives -- it is what Cycles
       computes, verified against a transcription of the kernel to 3e-12 -- and not on the
       ground that it fixed a brightness error, which it did not.

    Everything else is Cycles' line for line: FORWARD differences (`h_x - h_c`, not a
    central difference), `Rx`/`Ry`/`det` built from `normal_in` rather than from `Ng` (they
    differ whenever a Normal Map feeds this node, and then `det = r^2 * (Ng . N_in)` rather
    than `r^2`), `max(strength, 0)`, and the final `normalize(mix(N, N', strength))`.

    THE FALLBACK, AND WHY IT IS NOT SILENT. `duv_dx` is only populated where a ray carries
    differentials -- the camera hit. Mitsuba does not propagate them past the first bounce
    (Cycles carries an approximate widened differential), so at deeper vertices the radius
    falls back to a texel-sized world-space length, i.e. the old behaviour. Bump seen
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
        Fallback differencing step in UV, converted to a world-space radius and used only
        where the shading point carries no ray differentials. Default 0, meaning "derive
        it from the height texture's resolution".
    '''

    def __init__(self, props):
        TextureBase.__init__(self, props)
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

    def _orthonormals(self, n):
        """Cycles' `make_orthonormals` (`util/math_float3.h`), branch and all.

        The basis is arbitrary but it must be THE SAME arbitrary basis: `b = cross(n, a)`
        makes `(a, b, n)` right-handed, which is what fixes `sign(det)` to +1 for an
        unperturbed normal. A different-handed basis flips every bump in the scene.
        """
        a = mi.Vector3f(n.z - n.y, n.x - n.z, n.y - n.x)
        # `if (N.x != N.y || N.x != N.z)` takes the first form; the all-equal case is
        # degenerate for it (the cross product with (1,1,1) vanishes) and takes the second.
        degenerate = (n.x == n.y) & (n.x == n.z)
        a = dr.select(degenerate, mi.Vector3f(n.z - n.y, n.x + n.z, -n.y - n.x), a)
        a = dr.normalize(a)
        return a, dr.cross(n, a)

    def _radius(self, si):
        """Cycles' compacted differential: the scalar `0.5*(|dP.dx| + |dP.dy|)`.

        Where the shading point carries ray differentials this is the screen-space pixel
        footprint radius, which is what Cycles uses. Where it does not -- every vertex past
        the camera hit, since Mitsuba drops differentials there -- it degenerates to a
        texel-sized world-space length, chosen per lane rather than globally so a scene with
        both kinds of vertex gets each treated correctly.
        """
        dx, dy = mi.Vector2f(si.duv_dx), mi.Vector2f(si.duv_dy)
        have = (dr.abs(dx.x) + dr.abs(dx.y) + dr.abs(dy.x) + dr.abs(dy.y)) > 0.0
        dp_du, dp_dv = mi.Vector3f(si.dp_du), mi.Vector3f(si.dp_dv)
        r_diff = 0.5 * (dr.norm(dp_du * dx.x + dp_dv * dx.y) +
                        dr.norm(dp_du * dy.x + dp_dv * dy.y))
        r_texel = 0.5 * (dr.norm(dp_du) * self.step_u + dr.norm(dp_dv) * self.step_v)
        return dr.select(have, r_diff, r_texel)

    def _uv_of(self, dp_du, dp_dv, w):
        """The UV offset whose surface displacement is `w`, i.e. Cycles' `differential_dudv`.

        Cycles drops the least stable of the three world axes and applies Cramer's rule to
        what is left. The normal-equation solve below is the same answer wherever `w` lies
        in span(dp_du, dp_dv) -- which it does here, since `w` is built from a basis of the
        tangent plane -- and it needs no axis-selection branch.
        """
        a11, a12, a22 = dr.dot(dp_du, dp_du), dr.dot(dp_du, dp_dv), dr.dot(dp_dv, dp_dv)
        g = a11 * a22 - a12 * a12
        inv = dr.select(g != 0.0, dr.rcp(g), 0.0)
        b1, b2 = dr.dot(dp_du, w), dr.dot(dp_dv, w)
        return mi.Vector2f((a22 * b1 - a12 * b2) * inv, (a11 * b2 - a12 * b1) * inv)

    def _perturbed_normal(self, si, active):
        if self.has_normal:
            n = dr.normalize(si.to_world(
                mi.Normal3f(2.0 * self.normal.eval_3(si, active) - 1.0)))
        else:
            n = mi.Normal3f(si.sh_frame.n)

        fw = self.filter_width.eval_1(si, active)

        # Cycles' isotropic footprint: a scalar radius and an orthonormal pair built from
        # the GEOMETRIC normal -- not the true anisotropic screen differential. See the
        # class docstring; using the anisotropic one measurably over-perturbs.
        dp_du = mi.Vector3f(si.dp_du)
        dp_dv = mi.Vector3f(si.dp_dv)
        radius = self._radius(si)
        ex, ey = self._orthonormals(mi.Normal3f(si.n))
        dp_dx = ex * radius
        dp_dy = ey * radius

        duv_dx = self._uv_of(dp_du, dp_dv, dp_dx * fw)
        duv_dy = self._uv_of(dp_du, dp_dv, dp_dy * fw)

        # FORWARD differences at the offset points -- `h_x - h_c`, not a central difference.
        h_c = self._height_at(si, 0.0, 0.0, active)
        h_x = self._height_at(si, duv_dx.x, duv_dx.y, active)
        h_y = self._height_at(si, duv_dy.x, duv_dy.y, active)

        # `dp_dx` / `dp_dy` enter here UNSCALED by `fw`: Cycles applies the filter width to
        # the texture-coordinate offset (`tex_coord.h`) and, separately, to the first term
        # below (`displace.h`) -- never to the `dP` that builds `Rx` / `Ry` / `det`.
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

    def eval_color3(self, si, active):
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


mi.register_field('bump', lambda props: Bump(props))
