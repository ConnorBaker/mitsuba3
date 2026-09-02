# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This code is derived from Blender Cycles and was developed at Meta.

from __future__ import annotations  # Delayed parsing of type annotations

from typing import Tuple

import drjit as dr
import mitsuba as mi

from .common import *
from .lobes import *


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BlenderPrincipledBSDF(mi.BSDF):
    """
    Principled BSDF that matches Blender Cycles 4.5.

    Parameters:
        - base_color (Texture): The base color texture
        - roughness (Texture): The roughness texture
        - anisotropic (Texture): The anisotropic texture
        - anisotropic_rot (Texture): The anisotropic rotation texture in [0, 1]
        - eta (float): Main index of refraction
        - transmission (Texture): The specular transmission texture
        - sheen (Texture): The sheen texture
        - sheen_tint (Texture): The sheen tint texture
        - sheen_roughness (Texture): The sheen roughness texture
        - spec_ior_level (Texture): The specular IOR level texture to adjust specularity. 0.0 removes all reflections, 1.0 doubles the reflections.
        - spec_tint (Texture): The specular tint texture
        - metallic (Texture): The metallic texture
        - clearcoat (Texture): The clearcoat texture
        - clearcoat_roughness (Texture): The clearcoat gloss texture
        - clearcoat_ior (Texture): The clearcoat IOR texture
        - clearcoat_tint (Texture): The clearcoat tint texture
        - clearcoat_normalmap (Texture): Clearcoat normalmap texture (default: null)
        - normalmap (Texture): Base layer normal map (default: null)
        - alpha (Texture): Opacity texture (default: 1.0)
    """

    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        # Parameter definitions
        self.two_sided = props.get("two_sided", False)
        # HSR: Blender's Principled `distribution` enum (GGX / MULTI_GGX), which this
        # plugin previously had no notion of -- it implemented MULTI_GGX unconditionally.
        # Measured on a one-material furnace sphere against Cycles (constant white world,
        # saturated base, mean over the frame), mi/cy at a material that says GGX:
        #
        #     dielectric r=0.5  1.0018      metal r=0.5  1.0126      glass r=0.5  1.0179
        #     dielectric r=0.9  1.0052      metal r=0.9  1.0433      glass r=0.9  1.0969
        #
        # -- i.e. up to +9.7% too bright, always in the same direction, because the
        # multiple-scattering energy Cycles only adds under MULTI_GGX was being added
        # always. The same table at MULTI_GGX is 0.9927..1.0180, which is what this plugin
        # already reproduced and still must.
        #
        # Default True because Cycles' own default is MULTI_GGX: `PrincipledBsdfNode`'s
        # constructor sets `distribution = CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID`.
        self.multiscatter = props.get("multiscatter", True)
        self.base_color = props.get_texture("base_color", 0.5)
        self.diffuse_roughness = props.get_texture("diffuse_roughness", 0.0)
        self.roughness = props.get_texture("roughness", 0.5)
        self.has_specular = True
        self.spec_ior_level = props.get_texture("spec_ior_level", 0.5)
        self.has_anisotropic = is_active(props, "anisotropic")
        self.anisotropic = props.get_texture("anisotropic", 0.0)
        self.anisotropic_rot = props.get_texture("anisotropic_rot", 0.0)
        self.has_transmission = is_active(props, "transmission")
        self.transmission = props.get_texture("transmission", 0.0)
        self.has_sheen = is_active(props, "sheen")
        self.sheen = props.get_texture("sheen", 0.0)
        self.has_sheen_tint = is_active(props, "sheen_tint")
        # HSR: Blender's `Sheen Tint` default is WHITE, and the neutral value of a TINT is 1,
        # not 0. Defaulting it to 0 makes an absent property delete the lobe the caller just
        # asked for by setting `sheen` -- a silent fallback, and the same trap `spec_tint`
        # carries. The exporter always emits this property, so nothing that round-trips
        # through Blender is affected; what is affected is a hand-written dict.
        self.sheen_tint = props.get_texture("sheen_tint", 1.0)
        self.sheen_roughness = props.get_texture("sheen_roughness", 0.5)
        self.has_spec_tint = is_active(props, "spec_tint")
        self.spec_tint = props.get_texture("spec_tint", 0.0)
        self.has_metallic = is_active(props, "metallic")
        self.metallic = props.get_texture("metallic", 0.0)

        self.has_clearcoat = is_active(props, "clearcoat")
        self.clearcoat = props.get_texture("clearcoat", 0.0)
        self.clearcoat_roughness = props.get_texture("clearcoat_roughness", 0.0)
        self.clearcoat_ior = props.get_texture("clearcoat_ior", 1.5)
        self.clearcoat_tint = props.get_texture("clearcoat_tint", 1.0)
        self.has_clearcoat_normalmap = is_active(props, "clearcoat_normalmap")
        self.clearcoat_normalmap = props.get_texture("clearcoat_normalmap", 0.0)

        assert not (
            self.has_transmission and self.two_sided
        ), "Only materials without a specular transmission lobe can be two sided."

        # Cycles' per-material "Bump Map Correction"
        # (`SOCKET_BOOLEAN(use_bump_map_correction, ..., true)` in `scene/shader.cpp`),
        # defaulted ON exactly as Blender defaults it. See `_bump_shadowing`.
        self.bump_map_correction = bool(props.get("bump_map_correction", True))
        self.has_normalmap = is_active(props, "normalmap")
        self.normalmap = props.get_texture("normalmap", 0.0)

        self.has_alpha = is_active(props, "alpha")
        self.alpha = props.get_texture("alpha", 1.0)

        # Eta and specular has one to one correspondence, both of them can not be specified.
        assert not ("eta" in props.keys() and "specular" in props.keys())

        eta = props.get("eta", 1.5)
        # self.eta = 1 is not plausible for transmission
        if self.has_transmission and (eta == 1.0):
            eta = 1.001
        self.eta = mi.Float(eta)

        self._initialize_lobes()

        # Ensure pre-computed tables are fetched outside of rendering kernel
        if self.has_clearcoat or self.has_specular or self.has_sheen:
            fetch_table("ggx_gen_schlick_ior_s")

    def traverse(self, cb):
        F = mi.ParamFlags
        cb.put("base_color", self.base_color, F.Differentiable)
        cb.put("metallic", self.metallic, F.Differentiable)
        cb.put("clearcoat", self.clearcoat, F.Differentiable)
        cb.put("diffuse_roughness", self.diffuse_roughness, F.Differentiable)
        cb.put("clearcoat_roughness", self.clearcoat_roughness, F.Differentiable)
        cb.put("clearcoat_ior", self.clearcoat_ior, F.Differentiable)
        cb.put("clearcoat_tint", self.clearcoat_tint, F.Differentiable)
        cb.put("roughness", self.roughness, F.Differentiable | F.Discontinuous)
        cb.put("anisotropic", self.anisotropic, F.Differentiable)
        cb.put("anisotropic_rot", self.anisotropic_rot, F.Differentiable)
        cb.put("spec_tint", self.spec_tint, F.Differentiable)
        cb.put("spec_ior_level", self.spec_ior_level, F.Differentiable)
        cb.put("sheen", self.sheen, F.Differentiable)
        cb.put("sheen_tint", self.sheen_tint, F.Differentiable)
        cb.put("sheen_roughness", self.sheen_roughness, F.Differentiable)
        cb.put("transmission", self.transmission, F.Differentiable)
        cb.put("normalmap", self.normalmap, F.Differentiable)
        cb.put("eta", self.eta, F.Differentiable | F.Discontinuous)
        cb.put("alpha", self.alpha, F.Differentiable)

    def parameters_changed(self, keys):
        # Update the self.has_* attributes based on keys
        for k in keys:
            name = f"has_{k}"
            if hasattr(self, name):
                setattr(self, name, True)

        if "eta" in keys:
            # Eta = 1 is not plausible for transmission.
            self.eta[self.eta == 1.0] = 1.001

        self._initialize_lobes()

    def _initialize_lobes(self):
        F = mi.BSDFFlags

        self.components_mapping = dotdict()
        components = []

        # Diffuse reflection lobe
        components.append(F.DiffuseReflection | F.FrontSide)
        self.components_mapping.diffuse = len(components) - 1

        # Clearcoat lobe
        if self.has_clearcoat:
            components.append(F.GlossyReflection | F.FrontSide)
            self.components_mapping.clearcoat = len(components) - 1

        # Specular transmission lobe
        if self.has_transmission:
            f = (
                F.GlossyReflection
                | F.GlossyTransmission
                | F.FrontSide
                | F.BackSide
                | F.NonSymmetric
            )
            if self.has_anisotropic:
                f = f | F.Anisotropic
            components.append(f)
            self.components_mapping.trans_reflect = len(components) - 1
            self.components_mapping.trans_refract = len(components) - 1

        # Main specular reflection lobe
        if self.has_specular:
            f = F.GlossyReflection | F.FrontSide | F.BackSide
            if self.has_anisotropic:
                f = f | F.Anisotropic
            components.append(f)
            self.components_mapping.specular = len(components) - 1

        # A transmissive material is double-sided in Cycles even though it cannot be
        # wrapped in `twosided` here: `shader_setup_from_ray` flips `sd->N`/`sd->Ng` on
        # EVERY backfacing hit, so all of its closures shade on interior hits too. The
        # impls reproduce that with the unconditional mirror; the flags must agree.
        if self.two_sided or self.has_transmission:
            for i in range(len(components)):
                components[i] |= F.FrontSide | F.BackSide

        if self.has_alpha:
            f = F.Null | F.FrontSide | F.BackSide
            components.append(f)
            self.components_mapping.null = len(components) - 1

        flags = 0
        for c in components:
            flags |= c

        self.m_components = components
        self.m_flags = flags

        dr.make_opaque(self.eta)

    def fetch_attributes(
        self, si: mi.SurfaceInteraction3f, active: mi.Bool
    ) -> dotdict:
        """
        Fetch BSDF attributes for the current surface interaction
        """
        attr = dotdict()
        # Cycles' "filter glossy" floor for this shade point, set by the integrator.
        # Zero on any interaction the integrator did not touch, which makes every lobe
        # below identical to what it was.
        attr.min_alpha = si.min_alpha
        attr.diffuse_roughness = self.diffuse_roughness.eval_1(si, active)
        attr.diffuse = 1.0
        attr.base_color = self.base_color.eval(si, active)
        attr.sheen = self.sheen.eval_1(si, active) if self.has_sheen else mi.Float(0.0)
        attr.sheen_tint = (
            self.sheen_tint.eval(si, active)
            if self.has_sheen_tint
            else mi.UnpolarizedSpectrum(1.0)  # HSR: neutral tint is 1; see __init__
        )
        attr.sheen_roughness = (
            self.sheen_roughness.eval_1(si, active) if self.has_sheen else mi.Float(0.0)
        )
        attr.anisotropic = (
            self.anisotropic.eval_1(si, active)
            if self.has_anisotropic
            else mi.Float(0.0)
        )
        attr.anisotropic_rot = (
            self.anisotropic_rot.eval_1(si, active)
            if self.has_anisotropic
            else mi.Float(0.0)
        )
        attr.roughness = self.roughness.eval_1(si, active)
        attr.transmission = (
            self.transmission.eval_1(si, active)
            if self.has_transmission
            else mi.Float(0.0)
        )
        attr.spec_tint = (
            self.spec_tint.eval(si, active)
            if self.has_spec_tint
            else mi.UnpolarizedSpectrum(0.0)
        )
        attr.spec_ior_level = self.spec_ior_level.eval_1(si, active)
        attr.metallic = (
            self.metallic.eval_1(si, active) if self.has_metallic else mi.Float(0.0)
        )
        attr.clearcoat = (
            self.clearcoat.eval_1(si, active) if self.has_clearcoat else mi.Float(0.0)
        )
        attr.clearcoat_roughness = self.clearcoat_roughness.eval_1(si, active)
        attr.clearcoat_ior = self.clearcoat_ior.eval_1(si, active)
        attr.clearcoat_tint = self.clearcoat_tint.eval(si, active)
        attr.clearcoat_normal = (
            self.clearcoat_normalmap.eval_3(si, active)
            if self.has_clearcoat_normalmap
            else mi.Normal3f(0.5, 0.5, 1.0)
        )
        attr.normal = (
            mi.Normal3f(self.normalmap.eval_3(si, active))
            if self.has_normalmap
            else mi.Normal3f(0.5, 0.5, 1.0)
        )
        attr.eta = sanitize_eta(self.eta)
        attr.alpha = (
            dr.clip(self.alpha.eval_1(si, active), 0.0, 1.0)
            if self.has_alpha
            else mi.Float(1.0)
        )
        return attr

    def _lobe_weights(
        self,
        attr: dotdict,
        ctx: mi.BSDFContext,
        sh_wi: mi.Vector3f,
        wh: mi.Vector3f,
        cc_wi: mi.Vector3f,
        glass_eta: mi.Float,
    ) -> Tuple[dotdict, dotdict, dotdict]:
        """
        Calculate lobe weights, sampling weights and masks.

        Runs entirely in the MIRRORED space: the callers abs `si.wi.z` unconditionally
        before building `sh_wi`/`wh`/`cc_wi`, exactly as Cycles' closures only ever see
        the flipped `sd->N`. `glass_eta` is the side-corrected relative IOR
        (`bsdf->ior = backfacing ? 1/ior : ior` in `svm/closure.h`) -- the ONE quantity
        through which the glass lobes still know which side of the interface they are on.
        """
        weights = dotdict()
        weights.diffuse = mi.UnpolarizedSpectrum(attr.diffuse)
        # HSR: the tint belongs on the WEIGHT, not on the albedo estimate. It used to be
        # applied to `albedos.sheen` alone -- so it governed the layering budget and, through
        # it, the lobe's sampling weight, while `_eval_pdf_impl` added the sheen value
        # UNTINTED. Budget and value disagreed by exactly the tint: with the tint at its old
        # black default the lobe was never sampled (a black base under a sheen of 1 rendered
        # 0.000000 against Cycles' 0.067688) yet still contributed to `eval`, and with a
        # white tint the furnace read 1.006160 -- energy from nothing. On the weight it is
        # counted once, in both paths.
        weights.sheen = mi.UnpolarizedSpectrum(attr.sheen) * attr.sheen_tint
        weights.clearcoat = mi.UnpolarizedSpectrum(attr.clearcoat)
        weights.metallic = mi.UnpolarizedSpectrum(attr.metallic)
        weights.specular = mi.UnpolarizedSpectrum(1.0)
        weights.trans_reflect = mi.UnpolarizedSpectrum(attr.transmission)
        weights.trans_refract = mi.UnpolarizedSpectrum(attr.transmission)

        # Only lobes with non-zero weights are considered as active
        masks = dotdict({k: (dr.mean(v) > 0.0) for k, v in weights.items()})

        # NO SIDE MASK ON ANY LOBE'S BUDGET. There used to be one: a geometric
        # `front_side` test zeroed clearcoat/metallic/specular here (it replaced an
        # earlier SHADING-horizon test, whose removal was the strong-bump deficit fix --
        # the shading-horizon rejection lives on in `_eval_pdf_impl` as
        # `incoming_above_shading`, on evaluation alone). But a side mask on the budget
        # has no Cycles analogue at all: `shader_setup_from_ray` flips `sd->N`/`sd->Ng`
        # on every backfacing hit, so every closure is built and layered on the flipped
        # frame -- there IS no back side by the time the closures exist. The callers now
        # mirror unconditionally, so `sh_wi` here is always on the incident side and the
        # cascade below is Cycles' layering verbatim on both sides of a surface.
        # History (the measured strong-bump ladder, the metal-turned-Lambertian budget
        # bug, the interior-hit budget destruction and its stopgap re-route): git log.

        # Fresnel coefficient for the main specular. `glass_eta`, not `attr.eta`: this
        # split plays the role of the fresnel INSIDE Cycles' glass closure, which reads
        # the side-corrected `bsdf->ior`. In the mirrored space `dot(sh_wi, wh)` is
        # positive on interior hits, so `mi.fresnel`'s own negative-cosine flip can no
        # longer stand in for it.
        F_spec_dielectric = mi.fresnel(dr.dot(sh_wi, wh), glass_eta)[0]
        masks.specular &= F_spec_dielectric > 0.0

        # Masks lobes based on flags enabled in the BSDF context
        masks.diffuse &= ctx.is_enabled(+mi.BSDFFlags.DiffuseReflection)
        masks.sheen &= ctx.is_enabled(+mi.BSDFFlags.GlossyReflection)
        masks.clearcoat &= ctx.is_enabled(+mi.BSDFFlags.GlossyReflection)
        masks.metallic &= ctx.is_enabled(+mi.BSDFFlags.GlossyReflection)
        masks.specular &= ctx.is_enabled(+mi.BSDFFlags.GlossyReflection)
        masks.trans_reflect &= ctx.is_enabled(+mi.BSDFFlags.GlossyReflection)
        masks.trans_refract &= ctx.is_enabled(+mi.BSDFFlags.GlossyTransmission)

        if self.has_alpha:
            for k, v in masks.items():
                masks[k] &= ctx.component != mi.UInt32(self.components_mapping.null)
        masks.null = mi.Bool(ctx.is_enabled(+mi.BSDFFlags.Null))

        # Zero-out weights for lobes that are not active
        for k, v in weights.items():
            weights[k] = dr.select(masks[k], weights[k], 0.0)

        # Lobe albedos
        albedos = dotdict()

        # Compute lobe attenuation coefficients
        attenuation = mi.Float(1.0)

        # Ensure attenuation factor is in [0, 1]
        sanitize = lambda x: dr.select(dr.isfinite(x), dr.clip(x, 0, 1), 0)
        layering = lambda albedo, weight, attenuation: sanitize(
            attenuation * (1.0 - dr.max(albedo * weight / attenuation))
        )

        # Sheen lobe
        if self.has_sheen:
            # HSR: `microfacet_estimate_albedo(NONE, r0=0)` returns a flat 1.0, which is not
            # this lobe's albedo -- the real one peaks at 0.087214 and falls to 0 at normal
            # incidence. Budgeting 1.0 for it made an ACTIVE sheen tint delete the substrate.
            albedos.sheen = mi.UnpolarizedSpectrum(
                sheen_directional_albedo(mi.Frame3f.cos_theta(sh_wi))
            )
            attenuation = layering(albedos.sheen, weights.sheen, attenuation)

        # Clearcoat lobe
        if self.has_clearcoat:
            weights.clearcoat *= attenuation
            albedos.clearcoat = microfacet_estimate_albedo(
                MicrofacetFresnel.DIELECTRIC,
                cc_wi,
                attr.clearcoat_roughness,
                r0=None,
                eta=attr.clearcoat_ior,
                reflection=True,
                transmission=False,
            )
            attenuation = layering(albedos.clearcoat, weights.clearcoat, attenuation)

            # Clearcoat tint models absorption in the layer
            cc_cos_theta_i = mi.Frame3f.cos_theta(cc_wi)
            optical_depth = 1.0 / dr.sqrt(
                1.0
                - dr.square(1.0 / attr.clearcoat_ior) * (1 - dr.square(cc_cos_theta_i))
            )
            cc_tint = dr.power(attr.clearcoat_tint, optical_depth)
            attenuation *= dr.lerp(1.0, cc_tint, attr.clearcoat)

        if self.has_metallic:
            weights.metallic *= attenuation
            # HSR: spend the budget only where the lobe SURVIVES its mask. `masks.metallic
            # &= front_side` above already zeroed this lobe on the back side, but this line
            # used the raw attribute, so on a back-side hit the metallic share was deducted
            # from the budget and then handed to a lobe with weight zero -- annihilated.
            met_frac = dr.select(masks.metallic, attr.metallic, mi.Float(0.0))
            attenuation = sanitize(attenuation * (1.0 - met_frac))
            # HSR: multiple-scattering compensation, using the `ggx_E` / `ggx_Eavg` tables
            # this branch already ships and nothing read. Applied HERE, before
            # `sampling_weights` is derived from `weights`, so the lobe-selection
            # probabilities carry the same scale the evaluation does.
            if self.multiscatter:
                weights.metallic *= ggx_energy_compensation(
                    microfacet_table_roughness(attr.roughness, attr.anisotropic),
                    mi.Frame3f.cos_theta(sh_wi), r0=attr.base_color
                )

        # Transmission lobe
        if self.has_transmission:
            weights.trans_reflect *= attenuation
            weights.trans_refract *= attenuation
            albedos.trans_reflect = mi.UnpolarizedSpectrum(F_spec_dielectric)
            albedos.trans_refract = mi.UnpolarizedSpectrum(
                (1.0 - F_spec_dielectric) * dr.sqrt(attr.base_color)
            )
            # HSR: the SAME split on both sides of the interface. An interior hit used to
            # send the entire remaining budget to the transmission lobes (`trans_frac = 1`
            # on the back side) -- a stopgap for the era when the reflective lobes were
            # side-masked off and the `1 - transmission` remainder would otherwise have
            # been destroyed. Cycles has no such re-route: on a backfacing hit its
            # closures are all built on the flipped normal and the glass closure still
            # takes exactly `transmission_weight * weight`, with the remainder flowing to
            # specular and diffuse (`svm/closure.h`, `weight *= 1 - transmission_weight`
            # with no side test anywhere near it). With the unconditional mirror those
            # lobes now carry their share on interior hits, so the re-route is gone.
            attenuation = sanitize(attenuation * (1.0 - attr.transmission))
            # HSR: multiple-scattering compensation for the glass lobe, from `ggx_glass_E`.
            # Applied to BOTH halves because that table integrates the reflected and the
            # refracted albedo together -- they are one lobe as far as the energy is
            # concerned, and splitting the scale between them would be arbitrary.
            if self.multiscatter:
                _ms_glass = ggx_energy_compensation(
                    attr.roughness, mi.Frame3f.cos_theta(sh_wi), eta=attr.eta, glass=True
                )
                weights.trans_reflect *= _ms_glass
                weights.trans_refract *= _ms_glass

        # Specular lobe
        if self.has_specular:
            spec_r0 = eta_to_r0(attr.eta) * 2.0 * attr.spec_ior_level
            spec_eta = r0_to_eta(spec_r0)
            spec_eta[attr.eta < 1.0] = 1.0 / spec_eta
            spec_r0 *= attr.spec_tint

            weights.specular *= attenuation
            weights.specular[spec_eta == 1.0] = mi.UnpolarizedSpectrum(0.0)

            albedos.specular = microfacet_estimate_albedo(
                MicrofacetFresnel.GENERALIZED_SCHLICK,
                sh_wi,
                attr.roughness,
                r0=spec_r0,
                eta=spec_eta,
                reflection=True,
                transmission=False,
            )
            albedos.specular[attr.spec_ior_level == 0.0] = 0.0
            # HSR: THE SPECULAR LOBE'S SINGLE-SCATTERING LOSS, SPENT THE WAY CYCLES SPENDS IT.
            #
            # `microfacet_estimate_albedo` returns the roughness-aware FRESNEL reflectance,
            # while a single-scattering lobe only DELIVERS reflectance * E(roughness, mu).
            # The `reflectance * (1 - E)` difference has to go somewhere, and the choice is
            # not free -- it is the whole content of Blender's GGX / MULTI_GGX enum:
            #
            #   GGX       Cycles LOSES it. `bsdf_microfacet_estimate_albedo` carries no
            #             `energy_scale` term (checked in `kernel/closure/bsdf_microfacet.h`),
            #             so the substrate is attenuated by the full Fresnel reflectance and
            #             the lobe under-delivers. The scene is darker, and that is Cycles'
            #             documented single-scattering deficit rather than a bug to repair.
            #   MULTI_GGX Cycles RECOVERS it INTO THE SPECULAR LOBE, via
            #             `microfacet_ggx_preserve_energy`, and leaves the substrate budget
            #             alone.
            #
            # This branch used to do NEITHER: it multiplied the LAYERING budget by E, which
            # hands the missing energy DOWN to the diffuse substrate, unconditionally and in
            # both modes. That conserves energy, so a white-base furnace read correctly and
            # the defect hid there -- but it is wrong three ways at once. It pays the energy
            # out DIFFUSE instead of specular (wrong direction, wrong lobe), tinted by
            # `base_color` instead of by the Fresnel term (wrong colour), and it fires under
            # GGX where Cycles genuinely loses the energy (wrong mode). A black substrate
            # loses the recovery entirely, which is where the furnace cannot see it.
            #
            # MEASURED, on the Blender 4.1 splash scene at 128 spp / max_bounces 32, every
            # material overridden to one Principled BSDF (base 0.5) -- the arm that exposed
            # this. Mitsuba's response to the enum was ZERO at every roughness while Cycles'
            # rose monotonically, because both of Mitsuba's arms were doing the substrate
            # handoff:
            #
            #     roughness   mi GGX->MULTI_GGX   cy GGX->MULTI_GGX
            #        0.2           +0.026%             +0.368%
            #        0.5           +0.005%             +2.219%
            #        0.8           -0.014%             +4.714%
            #        1.0           +0.022%             +6.193%
            #
            # The metallic lobe already had this right (it compensates the WEIGHT, above),
            # which is why the same ladder's metal arm tracked Cycles to within 0.7 points
            # (+13.62% vs +14.36%) -- the dielectric lobe was the one still unfixed.
            if self.multiscatter:
                # Cycles' net scale on the lobe is `S = 1 + Fms * (1-E)/E` with
                # `Fms = Fss * E_avg / (1 - Fss * (1 - E_avg))`. `ggx_energy_compensation` is
                # the identical Kulla-Conty expression with `F_avg` in place of `Fss`, so
                # passing Cycles' own `Fss` as `r0` reproduces it rather than approximating.
                _S = ggx_energy_compensation(
                    microfacet_table_roughness(attr.roughness, attr.anisotropic),
                    mi.Frame3f.cos_theta(sh_wi),
                    r0=generalized_schlick_Fss(spec_r0, spec_eta),
                )
                # THE SPLIT MATTERS, AND GETTING IT WRONG COSTS THE WHOLE EFFECT.
                #
                # Cycles spends `S` in two places that are NOT interchangeable. It scales the
                # EVALUATION by `energy_scale = 1/E`, and it scales the stored WEIGHT by
                # `darkening = S*E` -- and only the second reaches the layering budget, because
                # the budget is `bsdf_albedo` = weight x estimate and `energy_scale` is applied
                # later, at eval. So the substrate is attenuated by `albedo * S * E` while the
                # lobe delivers `albedo * S`.
                #
                # Boosting the weight BEFORE `layering` instead makes the diffuse substrate
                # give up exactly what the specular lobe gains, and the render does not move.
                # Measured that way: scene gain +0.091% where Cycles reads +2.219% (roughness
                # 0.5, 128 spp, 32 bounces). The white furnace hid it -- it scored +0.008% --
                # while the BLACK-base control showed the full +2.48% at roughness 0.8, because
                # with `base_color` 0 there is no substrate for the boost to be stolen back by.
                # A control that only works on one of the two arms is the tell.
                albedos.specular *= _S * ggx_directional_albedo(
                    microfacet_table_roughness(attr.roughness, attr.anisotropic), mi.Frame3f.cos_theta(sh_wi)
                )
                attenuation = layering(albedos.specular, weights.specular, attenuation)
                weights.specular *= _S
            else:
                # Plain GGX: Cycles sets neither term, so the substrate is attenuated by the
                # full Fresnel reflectance while the lobe delivers only `reflectance * E`.
                # The difference is LOST, and that loss is what the enum exists to switch off.
                attenuation = layering(albedos.specular, weights.specular, attenuation)

        # Diffuse lobe
        weights.diffuse *= attenuation
        albedos.diffuse = mi.UnpolarizedSpectrum(attr.base_color)

        # Negative weights not allowed
        for k, v in weights.items():
            weights[k] = dr.maximum(v, 0.0)

        # Compute lobe sampling weights
        sampling_weights = dotdict()
        for k, v in weights.items():
            sampling_weights[k] = dr.mean(weights[k])

        # Account for lobe albedos in sampling weights
        for k, v in albedos.items():
            sampling_weights[k] *= dr.mean(albedos[k])

        # Normalize and detach the sampling weights
        normalization = dr.rcp(sum(sampling_weights.values()))
        for k, v in sampling_weights.items():
            sampling_weights[k] = dr.detach(v * normalization)

        if self.has_alpha:
            for k, v in sampling_weights.items():
                sampling_weights[k] *= attr.alpha
            sampling_weights.null = 1.0 - attr.alpha

        return weights, sampling_weights, masks

    def _valid_reflection_frame(self, si, frame, wiz0, attr, use_rot=True):
        """Cycles' `maybe_ensure_valid_specular_reflection` (`kernel/closure/bsdf_util.h`),
        as a FRAME for the glossy lobes -- piece 3 of the bump-map correction.

        If the closure normal would specularly reflect the view ray into the lower
        hemisphere of the GEOMETRIC surface, rotate it toward the geometric normal just
        far enough that the reflection clears a threshold (`min(0.9*dot(Ng,I), 0.01)`).
        Cycles applies the corrected normal to the metallic, specular and transmission
        microfacet closures ONLY -- the diffuse-range closures keep the uncorrected one
        (`svm/closure.h`: `bsdf->N = valid_reflection_N` on those three, plain `N` on
        Oren-Nayar) -- and the caller here mirrors that split. Wiring the whole frame
        through the existing `compute_frame(..., correction=True)` helper would move the
        diffuse lobe too, which is a different model; that is why this is per-lobe.

        FRAME BOOKKEEPING, which is where a faithful transcription can silently break:
        everything here lives in `si`'s local frame, where the SMOOTH shading normal is
        (0,0,1) and the geometric normal is `si.to_local(si.n)` -- NOT (0,0,1) on a
        smooth-shaded mesh, which is exactly why the correction also fires without any
        normal map, as it does in Cycles (`isequal(sd->Ng, N)` compares against the
        GEOMETRIC normal). Two conventions have to be reconciled:

        * the two-sided path mirrors `si.wi.z` about the tangent plane instead of
          flipping the frame, so the geometric normal must be mirrored the same way --
          `wiz0` carries the PRE-mirror sign of `si.wi.z` for that;
        * Cycles' `shader_setup_from_ray` flips `sd->Ng` onto the incident side of a
          backfacing hit (the `kernel_assert(Iz >= 0)` invariant). Flipping our mirrored
          `ng` to the hemisphere of the view direction reproduces that invariant for the
          one-sided interior-hit case (glass) as well.

        The quartic solve is the kernel's, term for term; the only additions are the
        `dr.maximum(..., eps)` guards on the two divisions, which Cycles leaves bare
        because its scalar path never evaluates the masked-out lanes.
        """
        n = mi.Vector3f(frame.n)

        ng = si.to_local(mi.Vector3f(si.n))
        # Mirror into the same reflected local space the unconditional mirror put
        # `si.wi` in (the mirror negates z only), then flip onto the incident side --
        # Cycles' backfacing flip of `sd->Ng`.
        ng.z = dr.mulsign(ng.z, wiz0)
        wi = si.wi
        ng = dr.mulsign(ng, dr.dot(ng, wi))

        r = 2.0 * dr.dot(n, wi) * n - wi
        iz = dr.dot(wi, ng)
        threshold = dr.minimum(0.9 * iz, 0.01)
        need = dr.dot(ng, r) < threshold

        # X: the component of N orthogonal to Ng, with N itself as the degenerate
        # fallback (`safe_normalize_fallback`).
        x_raw = n - dr.dot(n, ng) * ng
        x_len2 = dr.squared_norm(x_raw)
        x = dr.select(x_len2 > 0.0, x_raw * dr.rsqrt(dr.maximum(x_len2, 1e-30)), n)

        ix = dr.dot(wi, x)
        a = dr.maximum(dr.square(ix) + dr.square(iz), 1e-20)
        b = 2.0 * (a + iz * threshold)
        c = dr.square(threshold + iz)
        disc = dr.safe_sqrt(dr.square(b) - 4.0 * a * c)
        nz2 = dr.select(ix < 0.0, 0.25 * (b + disc) / a, 0.25 * (b - disc) / a)
        nx = dr.safe_sqrt(1.0 - nz2)
        nz = dr.safe_sqrt(nz2)
        n_new = nx * x + nz * ng

        n_out = dr.normalize(dr.select(need, n_new, n))

        # Rebuild the frame exactly as `compute_normalmap_frame` does, tangent included,
        # so the anisotropic rotation survives the correction.
        dp_du = mi.Vector3f(si.dp_du)
        if use_rot and attr.anisotropic_rot is not None:
            dp_du = mi.Transform4f().rotate(si.n, attr.anisotropic_rot * 360.0) @ dp_du
        dp_du = si.to_local(dp_du)
        g_frame = mi.Frame3f()
        g_frame.n = n_out
        g_frame.s = dr.normalize(-n_out * dr.dot(n_out, dp_du) + dp_du)
        g_frame.t = dr.cross(g_frame.n, g_frame.s)
        return g_frame

    def _bump_shadowing(self, si, frame, wo, is_eval):
        """Cycles' `bump_shadowing_term` (`kernel/closure/bsdf.h`).

        Returns `(keep, soft)`: a mask that rejects light leaking through the geometry, and
        a multiplicative shadowing factor. Both are identities wherever the closure normal
        equals the smooth one, so a material with no normal map pays nothing.

        WHY THIS EXISTS, MEASURED. Without it Mitsuba reproduces the render Cycles produces
        with `use_bump_map_correction` DISABLED, which is a different picture: on a plane
        with a constant tilted normal map under a single delta sun, mi/cy against the
        correction-OFF arm is 1.0000 at sun tilts of 89, 85, 80, 70 and 45 degrees, while
        against the ON arm it is 5.5884, 1.6272, 1.2080, 1.0559, 1.0078. The error is
        concentrated at grazing incidence and reaches 5.6x. A control arm with an
        unperturbed normal map matched at 1.0000 across the same sweep, so that gap is this
        term and not the scene.

        THE TWO HALVES HAVE DIFFERENT SCOPE, which is not a detail:

        * `keep` is a hemisphere-consistency test. `dot(Ns, I) * dot(Ns, N') * dot(N', I)`
          is negative exactly when `I` and `N'` sit on opposite sides of the SMOOTHED
          surface, i.e. when normal mapping has let light through the back of geometry that
          is supposed to be opaque. Cycles applies it to every closure on eval and to
          diffuse ones only when sampling -- so that a sampled specular direction is not
          killed outright -- and `is_eval` carries that distinction here.
        * `soft` is a GGX shadowing-masking term standing in for the microsurface the normal
          map implies, and Cycles applies it to DIFFUSE-range closures only. That range
          includes sheen (`CLOSURE_BSDF_SHEEN_ID` sits between `CLOSURE_BSDF_DIFFUSE_ID` and
          `CLOSURE_BSDF_TRANSLUCENT_ID`), which is why the caller applies it to the sheen
          lobe as well as the diffuse one.

        Everything is computed in `si`'s local frame, where the smooth normal `Ns` is
        exactly (0, 0, 1). That is not a shortcut: it removes a world-space round trip whose
        error would land on `cos_d`, and `cos_d` enters through `1/cos_d^2 - 1`, which is
        unbounded as the perturbation approaches 90 degrees.

        ONE DELIBERATE DIVERGENCE: Cycles also returns 1 for `PRIMITIVE_CURVE`, because a
        curve's `sd->N` is not a surface normal. This BSDF is never bound to a curve by the
        Blender exporter (hair goes to a different node), so the guard has no analogue here
        and is not stubbed in.

        Based on "A Microfacet-Based Shadowing Function to Solve the Bump Terminator
        Problem", Estevez, Lecocq and Stein.
        """
        # N' is ALREADY in `si`'s local frame: `compute_normalmap_frame` builds it as
        # `normalize(2 * normal - 1)`, a tangent-space vector, and returns `Frame3f([0,0,1])`
        # when there is no normal map. So it is used directly -- `si.to_local` here would
        # apply a second world-to-local transform and silently corrupt `cos_d`.
        #
        # This also settles the two-sided case without a flip. `_eval_pdf_impl` mirrors
        # `wo_.z` and `si.wi.z` about the tangent plane rather than flipping the frame, so a
        # back-face hit is evaluated in a reflected local space whose +z is the incident
        # side. A tangent-space normal map is defined relative to its own face, so `frame.n`
        # is already expressed in that same reflected space -- which is exactly the
        # assumption the surrounding code makes when it calls `frame.to_local(si.wi)`.
        n_c = mi.Vector3f(frame.n)

        cos_ns_i = wo.z                      # dot(Ns, I), Ns = (0, 0, 1)
        cos_ns_n = n_c.z                     # dot(Ns, N')
        cos_n_i = dr.dot(n_c, wo)            # dot(N', I)

        # `is_eval` is not consumed here: the kernel's `(is_eval || is_diffuse)` narrows
        # WHICH CLOSURES the rejection reaches, not the test itself, and the caller owns the
        # per-lobe masks. It stays in the signature so the call site reads like the kernel.
        keep = (cos_ns_i * cos_ns_n * cos_n_i) >= 0.0

        if not self.bump_map_correction:
            # The material turned the correction off. The rejection is part of the same
            # switch in Cycles (`SD_USE_BUMP_MAP_CORRECTION` gates the softening only, but
            # the rejection is reached first and is unconditional), so keep it and drop the
            # softening alone -- matching the kernel's control flow rather than the name.
            return keep, mi.Float(1.0)

        cos_i = dr.abs(cos_ns_i)
        cos_d = dr.abs(cos_ns_n)

        tan2_d = dr.rcp(dr.maximum(dr.square(cos_d), 1e-12)) - 1.0
        alpha2 = dr.clip(0.125 * tan2_d, 0.0, 1.0)
        tan2_i = dr.maximum(dr.rcp(dr.maximum(dr.square(cos_i), 1e-12)) - 1.0, 0.0)
        # bsdf_G<GGX>(alpha2, cos_N) = 1 / (1 + lambda), Heitz eq. 72
        lam = 0.5 * (dr.sqrt(1.0 + alpha2 * tan2_i) - 1.0)
        soft = dr.rcp(1.0 + lam)

        # Cycles' early-outs, in its order: an unperturbed or grazing-to-degenerate
        # configuration is left alone, and a vanishing `cos_i` is fully shadowed.
        soft = dr.select((cos_d >= 1.0) | (cos_i >= 1.0), mi.Float(1.0), soft)
        soft = dr.select(cos_i < 1e-6, mi.Float(0.0), soft)
        return keep, soft

    def _eval_pdf_impl(self, attr, ctx, si, wo_, active, is_eval=True, wiz0=None):
        # Two-sided
        wo_ = mi.Vector3f(wo_)
        si = mi.SurfaceInteraction3f(si)
        # `wiz0` is the PRE-mirror sign of `si.wi.z`, which the geometric normal below
        # needs. An integrator's eval/pdf call passes an unmirrored `si`, so it is
        # captured here; `_sample_impl` calls in with an `si` whose `wi.z` was ALREADY
        # abs'd, so it must pass its own pre-flip copy -- otherwise a
        # back-face sample's `bs.pdf` is computed in a differently-mirrored glossy
        # frame than the integrator's `pdf()` query for the same direction (MIS bias).
        if wiz0 is None:
            wiz0 = mi.Float(si.wi.z)
        # THE MIRROR IS UNCONDITIONAL, matching Cycles' `shader_setup_from_ray`, which
        # flips `sd->N` and `sd->Ng` on EVERY backfacing hit -- there is no un-flipped
        # shading in that engine. A transmissive material cannot be wrapped in
        # `twosided`, so its interior hits used to arrive here un-mirrored and every
        # reflective lobe died on its own side tests; the `1 - transmission` budget was
        # then re-routed to the glass lobes (`_lobe_weights`' old back-side override),
        # which is a different interior model than Cycles' (all closures alive on the
        # flipped frame). The glass lobes keep their inside/outside physics through
        # `glass_eta` below -- Cycles' `bsdf->ior = backfacing ? 1/ior : ior`
        # (`svm/closure.h`); the specular lobe's eta is NOT flipped there, and is not
        # flipped here.
        wo_.z = dr.mulsign(wo_.z, si.wi.z)
        si.wi.z = dr.abs(si.wi.z)
        glass_eta = dr.select(wiz0 >= 0.0, attr.eta, dr.rcp(attr.eta))

        # Apply normalmap
        frame = compute_normalmap_frame(si, attr.normal, rot=attr.anisotropic_rot)
        wi = frame.to_local(si.wi)
        wo = frame.to_local(wo_)

        # Compute half vector. In the mirrored space a refraction configuration has its
        # relative IOR inverted on interior hits, so the half-vector uses `glass_eta`.
        wh = half_vector(wi, wo, glass_eta)

        # Cycles' bump-map correction, piece 3: the GLOSSY lobes see a closure normal
        # raised so the specular reflection clears the geometric surface; the
        # diffuse-range lobes keep the uncorrected frame. See `_valid_reflection_frame`.
        if self.bump_map_correction:
            g_frame = self._valid_reflection_frame(si, frame, wiz0, attr)
        else:
            g_frame = frame
        wi_g = g_frame.to_local(si.wi)
        wo_g = g_frame.to_local(wo_)
        wh_g = half_vector(wi_g, wo_g, glass_eta)

        # Apply normalmap for clearcoat lobe. Cycles gives the coat its OWN copy of the
        # ensure-valid correction (`svm/closure.h`: `valid_coat_normal =
        # maybe_ensure_valid_specular_reflection(sd, coat_normal)`) and uses it both for
        # the coat closure and for the tint's optical depth (`cosNI = dot(sd->wi,
        # valid_coat_normal)`), so the corrected frame feeds everything downstream of
        # `cc_wi`. The coat is isotropic (`bsdf->T = zero_float3()`), hence use_rot=False.
        cc_frame = compute_normalmap_frame(si, normal=attr.clearcoat_normal)
        if self.bump_map_correction and self.has_clearcoat:
            cc_frame = self._valid_reflection_frame(si, cc_frame, wiz0, attr,
                                                    use_rot=False)
        cc_wi = cc_frame.to_local(si.wi)

        # Compute lobe weights and sample weights
        weights, sampling_weights, masks = self._lobe_weights(
            attr, ctx, wi, wh, cc_wi, glass_eta
        )

        # Shading and geometric horizon validity flags
        reflect_geom = is_reflection(si.wi, wo_)
        refract_geom = is_refraction(si.wi, wo_)
        reflect_shading = is_reflection(wi, wo)
        refract_shading = is_refraction(wi, wo)
        # The glossy lobes test their OWN (corrected) closure normal, exactly as Cycles'
        # `bsdf_microfacet_eval` tests `sc->N` after `bsdf->N = valid_reflection_N`.
        reflect_shading_g = is_reflection(wi_g, wo_g)
        refract_shading_g = is_refraction(wi_g, wo_g)

        # THE DIFFUSE-RANGE LOBES TEST ONLY THE OUTGOING DIRECTION against the closure
        # normal. Cycles' `bsdf_diffuse_eval` and `bsdf_sheen_eval` both declare their
        # incident direction as `const float3 /*wi*/` -- unnamed, unused -- and return
        # `max(dot(N', wo), 0)`; `bsdf_oren_nayar_eval` gates on `cosNO > 0` and CLAMPS the
        # view term rather than rejecting on it. Only the MICROFACET lobes reject
        # `cos_NI <= 0` (`bsdf_microfacet_eval`), so only they keep the two-direction test.
        #
        # THIS CHANGE ON ITS OWN IS A MEASURED NO-OP, and it is written down that way because
        # the opposite was nearly written down. Relaxing these two masks was tried FIRST, as
        # a fix for the strong-bump deficit, and the panel came back BIT-IDENTICAL: the
        # hypothesis was right about the mechanism and wrong about the site, because
        # `_lobe_weights` had already zeroed `weights.diffuse` and `weights.sheen` through
        # its own `&= front_side`, so a relaxed mask here had nothing left to admit. The
        # parity numbers belong to that fix and are recorded beside it. Both edits are
        # required and neither moves a pixel alone -- which is exactly why the no-op result
        # must not be read as "this line was already correct".
        #
        # `reflect_geom` stays. It is Cycles' smooth-surface condition, which that engine
        # applies through `bump_shadowing_term` rather than in the closure.
        outgoing_above_shading = mi.Frame3f.cos_theta(wo) > 0.0
        # EVERY microfacet closure tests the view against the closure normal:
        # `bsdf_microfacet_eval` opens with `cos_NI <= 0 -> zero_spectrum()`, commented
        # "Incoming direction has to be in the upper hemisphere (Cycles convention)", and it
        # is the shared entry for the reflective, refractive AND glass closures alike. That
        # test used to live in `_lobe_weights` as `&= front_side`, where it also silently
        # governed the layering budget; it is here now so it governs evaluation and nothing
        # else.
        #
        # NOT a bare `cos_theta(wi) > 0`. The Cycles convention holds because
        # `shader_setup_from_ray` FLIPS `sd->N` (and `sd->Ng`) on a backfacing hit, so
        # `cos_NI > 0` there describes the side the ray actually arrived from. Mitsuba keeps
        # the un-flipped frame for a one-sided BSDF, so the faithful transcription is the
        # SAME-SIGN test below: the closure normal and the geometric normal must agree about
        # which side the view is on. Spelling it `cos_theta(wi) > 0` would reject every
        # interior hit on a glass material -- which is not a normal-map case at all, and
        # would have been a much larger regression than the one being fixed.
        #
        # `reflect_shading` is not a substitute: it compares `wi` against `wo`, so it admits
        # the case where BOTH sit below the closure horizon on a geometrically front-facing
        # hit, which Cycles rejects outright.
        incoming_above_shading = (mi.Frame3f.cos_theta(wi)
                                  * mi.Frame3f.cos_theta(si.wi)) > 0.0
        incoming_above_shading_g = (mi.Frame3f.cos_theta(wi_g)
                                    * mi.Frame3f.cos_theta(si.wi)) > 0.0
        masks.diffuse &= reflect_geom & outgoing_above_shading
        masks.clearcoat &= reflect_geom & reflect_shading & incoming_above_shading
        masks.sheen &= reflect_geom & outgoing_above_shading
        masks.metallic &= reflect_geom & reflect_shading_g & incoming_above_shading_g
        masks.specular &= reflect_geom & reflect_shading_g & incoming_above_shading_g
        masks.trans_reflect &= reflect_geom & reflect_shading_g & incoming_above_shading_g
        masks.trans_refract &= refract_geom & refract_shading_g & incoming_above_shading_g

        # Cycles' bump-map correction. `bump_keep` rejects light leaking through the
        # smoothed geometry; `bump_soft` is the GGX shadowing factor, which Cycles applies
        # to DIFFUSE-RANGE closures only -- diffuse and sheen here. See `_bump_shadowing`.
        bump_keep, bump_soft = self._bump_shadowing(si, frame, wo_, is_eval)
        masks.diffuse &= bump_keep
        masks.sheen &= bump_keep
        if is_eval:
            # On eval the rejection covers every closure; when sampling it does not, so a
            # sampled specular direction survives. The glossy lobes' rejection tests
            # THEIR closure normal -- `bump_shadowing_term(sd, sc, ...)` reads `sc->N`,
            # which piece 3 just corrected -- so it comes from the corrected frame.
            bump_keep_g, _ = self._bump_shadowing(si, g_frame, wo_, is_eval)
            masks.clearcoat &= bump_keep
            masks.metallic &= bump_keep_g
            masks.specular &= bump_keep_g
            masks.trans_reflect &= bump_keep_g
            masks.trans_refract &= bump_keep_g

        # Initializing the output PDF and BSDF values
        pdf = mi.Float(0.0)
        value = mi.Spectrum(0.0)

        # Sheen evaluation
        if self.has_sheen:
            sheen_value = SheenLobe.eval(wi, wo, attr.sheen_roughness, attr.anisotropic)
            value[masks.sheen] += (mi.Spectrum(weights.sheen)
                                   * mi.Spectrum(sheen_value)
                                   * mi.Spectrum(bump_soft))

        # Clearcoat lobe
        if self.has_clearcoat:
            cc_value, cc_pdf = MicrofacetLobe.eval_pdf(
                wi,
                wo,
                reflection=True,
                roughness=attr.clearcoat_roughness,
                min_alpha=attr.min_alpha,
                fresnel_mode=MicrofacetFresnel.DIELECTRIC,
                eta=attr.clearcoat_ior,
                anisotropic=0.0,
                correlated_shadow_masking=True,
            )

            value[masks.clearcoat] += mi.Spectrum(weights.clearcoat) * mi.Spectrum(
                cc_value
            )
            pdf[masks.clearcoat] += sampling_weights.clearcoat * cc_pdf

        # Metallic component based on Schlick
        if self.has_metallic:
            metal_value, metal_pdf = MicrofacetLobe.eval_pdf(
                wi_g,
                wo_g,
                reflection=True,
                roughness=attr.roughness,
                min_alpha=attr.min_alpha,
                anisotropic=attr.anisotropic,
                fresnel_mode=MicrofacetFresnel.APPROXIMATED_SCHLICK,
                r0=attr.base_color,
                eta=attr.eta,
                correlated_shadow_masking=False,
            )

            value[masks.metallic] += mi.Spectrum(weights.metallic) * mi.Spectrum(
                metal_value
            )
            pdf[masks.metallic] += sampling_weights.metallic * metal_pdf

        # Glass lobe. `eta=glass_eta` is the side-corrected IOR -- Cycles'
        # `bsdf->ior = backfacing ? 1/ior : ior` -- which its fresnel ALSO reads
        # (`microfacet_fresnel` passes `bsdf->ior` into the real-Fresnel curve).
        # `r0` stays on `attr.eta`: Cycles builds `fresnel->f0` from the UN-flipped ior
        # (`generalized_schlick_setup(ior, ...)`), and `F0_from_ior` is reciprocal-
        # symmetric anyway, so the two spellings agree.
        if self.has_transmission:
            reflect_value, reflect_pdf = MicrofacetLobe.eval_pdf(
                wi_g,
                wo_g,
                reflection=True,
                roughness=attr.roughness,
                min_alpha=attr.min_alpha,
                anisotropic=attr.anisotropic,
                fresnel_mode=MicrofacetFresnel.GENERALIZED_SCHLICK,
                eta=glass_eta,
                r0=eta_to_r0(attr.eta) * attr.spec_tint,
                correlated_shadow_masking=True,
            )
            value[masks.trans_reflect] += mi.Spectrum(
                weights.trans_reflect
            ) * mi.Spectrum(reflect_value)
            pdf[masks.trans_reflect] += sampling_weights.trans_reflect * reflect_pdf

            refract_value, refract_pdf = MicrofacetLobe.eval_pdf(
                wi_g,
                wo_g,
                reflection=False,
                roughness=attr.roughness,
                min_alpha=attr.min_alpha,
                anisotropic=attr.anisotropic,
                fresnel_mode=MicrofacetFresnel.GENERALIZED_SCHLICK,
                eta=glass_eta,
                r0=eta_to_r0(attr.eta) * attr.spec_tint,
                correlated_shadow_masking=True,
            )
            value[masks.trans_refract] += (
                mi.Spectrum(weights.trans_refract)
                * mi.Spectrum(refract_value)
                * mi.Spectrum(dr.sqrt(attr.base_color))
            )
            pdf[masks.trans_refract] += sampling_weights.trans_refract * refract_pdf

        # Specular reflection lobe
        if self.has_specular:
            spec_r0 = eta_to_r0(attr.eta) * 2.0 * attr.spec_ior_level
            spec_eta = r0_to_eta(spec_r0)
            spec_eta[attr.eta < 1.0] = 1.0 / spec_eta
            spec_r0 *= attr.spec_tint

            spec_value, spec_pdf = MicrofacetLobe.eval_pdf(
                wi_g,
                wo_g,
                reflection=True,
                roughness=attr.roughness,
                min_alpha=attr.min_alpha,
                anisotropic=attr.anisotropic,
                fresnel_mode=MicrofacetFresnel.GENERALIZED_SCHLICK,
                r0=spec_r0,
                eta=spec_eta,
            )

            value[masks.specular] += mi.Spectrum(weights.specular) * mi.Spectrum(
                spec_value
            )
            pdf[masks.specular] += sampling_weights.specular * spec_pdf

        if True:
            # Adding diffuse lobe
            diffuse_value, diffuse_pdf = OrenNayarLobe.eval_pdf(
                wi, wo, attr.base_color, attr.diffuse_roughness
            )
            value[masks.diffuse] += (
                mi.Spectrum(diffuse_value)
                * mi.Spectrum(weights.diffuse)
                * mi.Spectrum(attr.base_color)
                * mi.Spectrum(bump_soft)
            )
            pdf[masks.diffuse] += sampling_weights.diffuse * diffuse_pdf

        if self.has_alpha:
            value *= mi.Spectrum(attr.alpha)
            # Lobe's pdfs are already multiplied by alpha in the sampling weights

        return value, dr.detach(pdf)

    def _sample_impl(self, attr, ctx, si, sample1, sample2, active):
        # Avoid modifying caller's state
        active = mi.Bool(active)

        null_wi = mi.Vector3f(si.wi)

        # Unconditional mirror -- see the twin comment in `_eval_pdf_impl`. `null_wi`
        # keeps the pre-mirror direction: its z sign is the true geometric side.
        si = mi.SurfaceInteraction3f(si)
        si.wi.z = dr.abs(si.wi.z)
        glass_eta = dr.select(null_wi.z >= 0.0, attr.eta, dr.rcp(attr.eta))

        # Apply normalmap
        frame = compute_normalmap_frame(si, attr.normal, rot=attr.anisotropic_rot)
        wi = frame.to_local(si.wi)

        # Piece 3 of the bump-map correction: the glossy lobes sample in the CORRECTED
        # frame, the same one `_eval_pdf_impl` evaluates them in -- eval and sample must
        # share a frame or the returned pdf stops describing the sampling distribution.
        # `null_wi` is the pre-mirror copy, so its z carries the sign the geometric
        # normal needs. See `_valid_reflection_frame`.
        if self.bump_map_correction:
            g_frame = self._valid_reflection_frame(si, frame, null_wi.z, attr)
        else:
            g_frame = frame
        wi_g = g_frame.to_local(si.wi)

        # Sample main specular and transmission reflection distribution. `wh` (base
        # frame, from the uncorrected `wi`) feeds `_lobe_weights` exactly as eval's
        # half-vector `wh` does; `wh_g` builds the actual sampled directions.
        wh = MicrofacetLobe.sample_wh(
            wi, sample2, attr.roughness, attr.anisotropic, min_alpha=attr.min_alpha
        )
        wh_g = MicrofacetLobe.sample_wh(
            wi_g, sample2, attr.roughness, attr.anisotropic, min_alpha=attr.min_alpha
        )

        # Apply normalmap for clearcoat lobe -- corrected exactly as in eval; sample and
        # eval must share the coat frame or the pdf stops describing the distribution.
        cc_frame = compute_normalmap_frame(si, normal=attr.clearcoat_normal)
        if self.bump_map_correction and self.has_clearcoat:
            cc_frame = self._valid_reflection_frame(si, cc_frame, null_wi.z, attr,
                                                    use_rot=False)
        cc_wi = cc_frame.to_local(si.wi)

        # Compute lobe weights and sample weights
        weights, sampling_weights, _ = self._lobe_weights(
            attr, ctx, wi, wh, cc_wi, glass_eta
        )

        # Pick lobe based on weights
        masks = dotdict()
        masks.diffuse = mi.Bool(active)
        masks.clearcoat = mi.Bool(active) & self.has_clearcoat
        masks.metallic = mi.Bool(active)
        masks.specular = mi.Bool(active)
        masks.trans_reflect = mi.Bool(active) & self.has_transmission
        masks.trans_refract = mi.Bool(active) & self.has_transmission
        masks.sheen = mi.Bool(active) & self.has_sheen
        masks.null = mi.Bool(active) & self.has_alpha

        cum = mi.Float(0.0)
        for k, w in sampling_weights.items():
            masks[k] &= (sample1 >= cum) & (sample1 < cum + w)
            cum += w

        # BSDF sampling data-structure to fill in
        bs = dr.zeros(mi.BSDFSample3f)

        # Diffuse lobe sampling
        if True:
            b = dr.zeros(mi.BSDFSample3f)
            b.wo = frame.to_world(OrenNayarLobe.sample(sample2))
            b.sampled_component = self.components_mapping.diffuse
            b.sampled_type = +mi.BSDFFlags.DiffuseReflection
            b.eta = 1.0
            bs[masks.diffuse & is_reflection(si.wi, b.wo)] = b

        # Clearcoat reflection sampling
        if self.has_clearcoat:
            cc_wh = MicrofacetLobe.sample_wh(
                cc_wi, sample2, attr.clearcoat_roughness, min_alpha=attr.min_alpha
            )
            b = dr.zeros(mi.BSDFSample3f)
            b.wo = frame.to_world(microfacet_reflect(cc_wi, cc_wh))
            b.sampled_component = self.components_mapping.clearcoat
            b.sampled_type = +mi.BSDFFlags.GlossyReflection
            b.eta = 1.0
            bs[masks.clearcoat & is_reflection(si.wi, b.wo)] = b

        # Reflection
        if self.has_specular:
            b = dr.zeros(mi.BSDFSample3f)
            b.wo = g_frame.to_world(microfacet_reflect(wi_g, wh_g))
            b.sampled_component = self.components_mapping.specular
            b.sampled_type = +mi.BSDFFlags.GlossyReflection
            b.eta = 1.0
            valid = is_reflection(si.wi, b.wo)
            bs[(masks.specular | masks.metallic | masks.sheen) & valid] = b

        # Transmission
        if self.has_transmission:
            # Reflection
            b = dr.zeros(mi.BSDFSample3f)
            b.wo = g_frame.to_world(microfacet_reflect(wi_g, wh_g))
            b.sampled_component = self.components_mapping.trans_reflect
            b.sampled_type = +mi.BSDFFlags.GlossyReflection
            b.eta = 1.0
            bs[masks.trans_reflect] = b

            # Refraction
            b = dr.zeros(mi.BSDFSample3f)
            # `glass_eta`, not a side test on `wi_g`: in the mirrored space `wi_g` is
            # ALWAYS on the front side, so a test on it can never flip -- the true
            # geometric side lives in `null_wi.z`, which `glass_eta` was built from.
            b.wo = g_frame.to_world(microfacet_refract(wi_g, wh_g, glass_eta))
            b.sampled_component = self.components_mapping.trans_refract
            b.sampled_type = +mi.BSDFFlags.GlossyTransmission
            b.eta = mi.Float(glass_eta)
            bs[masks.trans_refract & is_refraction(si.wi, b.wo)] = b

        if self.has_alpha:
            b = dr.zeros(mi.BSDFSample3f)
            b.wo = -null_wi
            b.sampled_component = self.components_mapping.null
            b.sampled_type = +mi.BSDFFlags.Null
            b.eta = 1.0
            bs[masks.null] = b

        # Compute corresponding PDF and BSDF value
        # `is_eval=False`: Cycles does not reject a SAMPLED non-diffuse direction,
        # only an evaluated one (`bsdf_sample` passes `false`).
        value, bs.pdf = self._eval_pdf_impl(attr, ctx, si, bs.wo, active,
                                            is_eval=False, wiz0=null_wi.z)

        if self.has_alpha:
            value[masks.null] = 1.0 - attr.alpha
            bs.pdf[masks.null] = dr.detach(1.0 - attr.alpha)

        active &= bs.pdf > 0.0

        # Compute sampling weight
        weight = dr.select(active, value / bs.pdf, 0.0)

        # Un-mirror. `si.wi.z` is NOT the sign to restore here: it was already
        # overwritten with `dr.abs(si.wi.z)` above, so `mulsign` against it can only ever
        # multiply by +1 and this flip-back silently does nothing. `null_wi` is the
        # pre-flip copy and carries the original sign. `_eval_pdf_impl` performs the same
        # two steps in the OPPOSITE order (mulsign first, then abs), which is why eval/pdf
        # were correct while sample was not -- and why `bsdf.pdf(bs.wo)` returns 0 on
        # exactly the samples this line failed to flip. The null lobe's direction was
        # built from the un-mirrored `null_wi` and must not be flipped.
        bs.wo.z[~masks.null] = dr.mulsign(bs.wo.z, null_wi.z)

        return bs, weight

    def eval(self, ctx, si, wo, active=True):
        attr = self.fetch_attributes(si, active)
        return self._eval_pdf_impl(attr, ctx, si, wo, active)[0]

    def pdf(self, ctx, si, wo, active=True):
        attr = self.fetch_attributes(si, active)
        return self._eval_pdf_impl(attr, ctx, si, wo, active)[1]

    def eval_pdf(self, ctx, si, wo, active=True):
        attr = self.fetch_attributes(si, active)
        return self._eval_pdf_impl(attr, ctx, si, wo, active)

    def sample(self, ctx, si, sample1, sample2, active):
        attr = self.fetch_attributes(si, active)
        return self._sample_impl(attr, ctx, si, sample1, sample2, active)

    def eval_null_transmission(self, si, active):
        if self.has_alpha:
            return 1.0 - dr.clip(self.alpha.eval_1(si, active), 0.0, 1.0)
        else:
            return 0.0

    def to_string(self):
        return (
            "Principled BSDF[\n"
            "    two_sided=%s,\n"
            "    base_color=%s,\n"
            "    transmission=%s,\n"
            "    roughness=%s,\n"
            "    anisotropic=%s,\n"
            "    eta=%s,\n"
            "    sheen=%s,\n"
            "    sheen_tint=%s,\n"
            "    sheen_roughness=%s,\n"
            "    clearcoat=%s,\n"
            "    clearcoat_roughness=%s,\n"
            "    clearcoat_ior=%s,\n"
            "    clearcoat_tint=%s,\n"
            "    clearcoat_normalmap=%s,\n"
            "    metallic=%s,\n"
            "    spec_tint=%s,\n"
            "    spec_ior_level=%s,\n"
            "    normalmap=%s,\n"
            "]"
            % (
                self.two_sided,
                self.base_color,
                self.transmission,
                self.roughness,
                self.anisotropic,
                self.eta,
                self.sheen,
                self.sheen_tint,
                self.sheen_roughness,
                self.clearcoat,
                self.clearcoat_roughness,
                self.clearcoat_ior,
                self.clearcoat_tint,
                self.clearcoat_normalmap,
                self.metallic,
                self.spec_tint,
                self.spec_ior_level,
                self.normalmap,
            )
        )


mi.register_bsdf("blender_principled", lambda props: BlenderPrincipledBSDF(props))
