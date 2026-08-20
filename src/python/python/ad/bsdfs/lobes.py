# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This code is derived from Blender Cycles and was developed at Meta.

from __future__ import annotations  # Delayed parsing of type annotations

from enum import Enum
from typing import Tuple

import drjit as dr
import mitsuba as mi

from .common import half_vector
from .tables import fetch_table


# ------------------------------------------------------------------------------
# Microfacet routines
# ------------------------------------------------------------------------------


def r0_to_eta(r0: mi.Float) -> mi.Float:
    """
    Convert R0 (reflectivity at perpendicular) into IOR
    """
    assert r0 is not None
    # Clamp the value of r0 in case it is unrealistic
    r0 = dr.clip(r0, 0.0, 0.3)
    sqrt_r0 = dr.sqrt(r0)
    return (1.0 + sqrt_r0) / (1.0 - sqrt_r0)


def eta_to_r0(eta: mi.Float) -> mi.Float:
    """
    Convert IOR into R0 (reflectivity at perpendicular)
    """
    assert eta is not None
    return dr.square((eta - 1.0) / (eta + 1.0))


def schlick_weight(cos_theta: mi.Float) -> mi.Float:
    """
    Computes the schlick weight for Fresnel Schlick approximation
    """
    m = dr.clip(1.0 - cos_theta, 0.0, 1.0)
    return dr.square(dr.square(m)) * m


# Enumeration of the microfacet fresnel modes
class MicrofacetFresnel(Enum):
    NONE = 0
    DIELECTRIC = 1
    CONDUCTOR = 2
    APPROXIMATED_SCHLICK = 3
    GENERALIZED_SCHLICK = 4


def fresnel_dielectric(cos_theta_i: mi.Float, eta: mi.Float):
    """
    Dielectric fresnel curve
    """
    assert eta is not None

    eta_cos_theta_t_sq = dr.square(eta) - (1.0 - dr.square(cos_theta_i))

    cos_theta_i = dr.abs(cos_theta_i)

    # Relative to the surface normal.
    cos_theta_t = -dr.safe_sqrt(eta_cos_theta_t_sq) / eta

    # Amplitudes of reflected waves.
    r_s = (cos_theta_i + eta * cos_theta_t) / (cos_theta_i - eta * cos_theta_t)
    r_p = (cos_theta_t + eta * cos_theta_i) / (cos_theta_t - eta * cos_theta_i)

    return dr.select(
        eta_cos_theta_t_sq > 0, 0.5 * (dr.square(r_s) + dr.square(r_p)), 1.0
    )


def fresnel_conductor(cos_theta: mi.Float, eta: mi.Float):
    """
    Conductor fresnel curve
    """
    assert eta is not None
    return mi.fresnel_conductor(cos_theta, eta)


def fresnel_generalized_schlick(
    cos_theta: mi.Float, r0: mi.Float, eta: mi.Float, r90: mi.Float = 1.0
) -> mi.Float:
    """
    Use real Fresnel curve to determine the interpolation between r0 and r90.
    """
    assert r0 is not None
    if eta is None:
        eta = r0_to_eta(r0)

    F_real = fresnel_dielectric(cos_theta, eta)
    F0 = eta_to_r0(eta)
    Fh = (F_real - F0) / (1.0 - F0)
    return dr.lerp(r0, r90, dr.maximum(Fh, 0.0))


def fresnel_schlick_approx(
    cos_theta: mi.Float, r0: mi.Spectrum, eta: mi.Float
) -> mi.Spectrum:
    """
    Schlick Approximation for Fresnel Reflection coefficient: F = r0 + (1 - r0) (1-cos_theta^5)
    """
    assert r0 is not None
    assert eta is not None

    is_front = cos_theta > 0.0
    eta_it = dr.select(is_front, eta, dr.rcp(eta))
    eta_ti = dr.select(is_front, dr.rcp(eta), eta)

    cos_theta_t = dr.safe_sqrt(
        dr.fma(-dr.fma(-cos_theta, cos_theta, 1.0), dr.square(eta_ti), 1.0)
    )

    return dr.select(
        eta_it > 1.0,
        dr.lerp(schlick_weight(dr.abs(cos_theta)), 1.0, r0),
        dr.lerp(schlick_weight(cos_theta_t), 1.0, r0),
    )


def microfacet_fresnel(
    fresnel_mode: MicrofacetFresnel, cos_theta: mi.Float, r0: mi.Float, eta: mi.Float
) -> mi.Float:
    """
    Dispatches to the appropriate Fresnel computation based on the selected microfacet Fresnel mode.
    """
    if fresnel_mode == MicrofacetFresnel.NONE:
        return 1.0
    elif fresnel_mode == MicrofacetFresnel.DIELECTRIC:
        return fresnel_dielectric(cos_theta, eta)
    elif fresnel_mode == MicrofacetFresnel.CONDUCTOR:
        return fresnel_conductor(cos_theta, eta)
    elif fresnel_mode == MicrofacetFresnel.GENERALIZED_SCHLICK:
        return fresnel_generalized_schlick(cos_theta, r0, eta)
    elif fresnel_mode == MicrofacetFresnel.APPROXIMATED_SCHLICK:
        return fresnel_schlick_approx(cos_theta, r0, eta)
    else:
        raise Exception("Invalid fresnel mode!")


def microfacet_compute_alphas(
    roughness: mi.Float, anisotropic: mi.Float, min_alpha: mi.Float = 0.0
) -> Tuple[mi.Float, mi.Float]:
    """
    Calculates the microfacet distribution parameters based on Disney Course Notes

    `min_alpha` is Cycles' "filter glossy" roughness floor, carried on the surface
    interaction (see `SurfaceInteraction.min_alpha`). It is applied to the FINAL alphas
    rather than to `roughness`, because that is where Cycles applies it:
    `bsdf_microfacet_blur` sets `alpha_x = max(roughness, alpha_x)` and likewise for
    `alpha_y`, i.e. AFTER the anisotropic aspect split. Clamping `roughness` instead
    would leave the two axes floored at different values. Zero is the identity.
    """
    roughness_2 = dr.square(roughness)
    aspect = dr.sqrt(1.0 - 0.9 * anisotropic)
    return dr.maximum(dr.maximum(1e-4, min_alpha), roughness_2 / aspect), dr.maximum(
        dr.maximum(1e-4, min_alpha), roughness_2 * aspect
    )


def microfacet_shadow_masking(
    distr: mi.MicrofacetDistribution,
    wi: mi.Vector3f,
    wo: mi.Vector3f,
    wh: mi.Vector3f,
    correlated_shadow_masking: bool = True,
) -> Tuple[mi.Float, mi.Float]:
    """
    Calculates the shadow masking term
    """
    if correlated_shadow_masking:
        # 2nd argument disable Mitsuba check for consistent orientation
        Gi = distr.smith_g1(wi, [0, 0, 1])
        Go = distr.smith_g1(wo, [0, 0, 1])

        # Smith's height correlated shadowing-masking function
        lambda_wi = dr.rcp(Gi) - 1.0
        lambda_wo = dr.rcp(Go) - 1.0
        G = dr.rcp(1.0 + lambda_wi + lambda_wo)
    else:
        Gi = distr.smith_g1(wi, wh)
        Go = distr.smith_g1(wo, wh)
        G = Gi * Go
    return G, Gi


def microfacet_reflect(wi: mi.Vector3f, wh: mi.Vector3f) -> mi.Vector3f:
    """
    Compute reflection direction based on incoming direction and half-vector
    """
    return mi.reflect(wi, wh)


def microfacet_refract(wi: mi.Vector3f, wh: mi.Vector3f, eta: mi.Float) -> mi.Vector3f:
    """
    Compute refraction direction based on incoming direction and half-vector
    """
    _, cos_theta_t, _, eta_ti = mi.fresnel(dr.dot(wi, wh), eta)
    return mi.refract(wi, wh, cos_theta_t, eta_ti)


def microfacet_estimate_albedo(
    fresnel_mode: MicrofacetFresnel,
    wi: mi.Vector3f,
    roughness: mi.Float,
    r0: mi.Float,
    eta: mi.Float,
    reflection: bool,
    transmission: bool,
    reflection_tint: mi.Spectrum = 1.0,
) -> mi.Spectrum:
    """
    This function estimates the albedo of the BSDF as caused by applied Fresnel
    model for the given view direction.
    """
    cos_theta_i = mi.Frame3f.cos_theta(wi)
    F = microfacet_fresnel(fresnel_mode, cos_theta_i, r0, eta)

    albedo = mi.Spectrum(0.0)

    if reflection:
        reflectance = F
        if fresnel_mode == MicrofacetFresnel.GENERALIZED_SCHLICK:
            # Reparameterize eta to z
            z = dr.sqrt(dr.abs((eta - 1.0) / (eta + 1.0)))
            s = fetch_table("ggx_gen_schlick_ior_s").eval([roughness, cos_theta_i, z])[
                0
            ]
            reflectance = dr.lerp(r0, 1.0, s) * reflection_tint
        albedo += mi.Spectrum(reflectance)

    if transmission:
        transmittance = 1.0 - F
        albedo += mi.Spectrum(transmittance)

    return mi.unpolarized_spectrum(albedo)


# ------------------------------------------------------------------------------


def sheen_directional_albedo(cos_theta: mi.Float) -> mi.Float:
    """
    Directional albedo of `SheenLobe`, i.e. the integral of its `eval` over the hemisphere.

    Needed because the layering code used to budget for this lobe with
    `microfacet_estimate_albedo(NONE, r0=0)`, which returns a flat 1.0 -- between 15x and
    1300x the truth, since the real albedo runs from 0.087214 at grazing down to ~0 at normal
    incidence. A layer that claims all of the incident energy and then delivers a fraction of
    a percent of it does not tint the surface below, it DELETES it. Measured on a white base
    at roughness 0.8 in a constant environment of radiance 1, with `sheen_tint` present and
    white -- which is what `mitsuba-blender`'s exporter writes for Blender's default:

        sheen weight     Cycles      this plugin
            0.2         0.999664      0.887187     (-11.2515%)
            1.0         0.999690      0.005587     (-99.4411%)

    The lobe accepts a `roughness` argument and never reads it, so this is a function of
    cos_theta alone -- checked across a 64-point grid, where E(roughness=0) and E(roughness=1)
    agree to 0.000e+00 exactly. That is why a cubic suffices and no new table is needed: it is
    a least-squares fit in (1 - mu) to a 2^23-sample hemisphere integration of the lobe, with
    no constant term because E(mu = 1) is 0 analytically (a normal-incidence retro direction
    gives cos_d = 1, where the Schlick weight vanishes). Max absolute error over the grid is
    3.561e-04, 0.408% of the peak.

    NOTE this fixes the plugin's ENERGY BOOKKEEPING, not its agreement with Cycles' sheen
    MODEL. `SheenLobe` is the Disney/Blender-3.x Schlick-weight sheen; Blender 4.x uses
    Zeltner et al.'s linearly-transformed-cosine microfibre sheen, which is roughness-
    dependent and carries roughly 5x this lobe's albedo. That gap is a separate, larger piece
    of work and is deliberately NOT claimed to be closed here.
    """
    t = 1.0 - dr.clip(dr.abs(cos_theta), 0.0, 1.0)
    return dr.maximum(
        t * (0.013357388 + t * (-0.004919108 + t * 0.079626067)), 0.0
    )


def ggx_directional_albedo(roughness: mi.Float, cos_theta: mi.Float) -> mi.Float:
    """
    Directional albedo E(roughness, mu) of a GGX lobe with the Fresnel term set to 1 --
    the FRACTION of the incident energy a SINGLE-SCATTERING lobe actually delivers.

    This is `ggx_E`, the same table `ggx_energy_compensation` divides by, exposed on its own
    because one caller needs the fraction rather than its reciprocal: a layered lobe has to
    subtract from the layer below what it DELIVERS, not what its Fresnel term promises.
    """
    return fetch_table("ggx_E").eval(
        [dr.clip(roughness, 1e-4, 1.0), dr.clip(dr.abs(cos_theta), 1e-4, 1.0)]
    )[0]


def ggx_energy_compensation(
    roughness: mi.Float,
    cos_theta: mi.Float,
    r0: mi.Spectrum = None,
    eta: mi.Float = None,
    glass: bool = False,
) -> mi.Spectrum:
    """
    Multiple-scattering energy compensation for a GGX lobe, Turquin-style.

    A single-scattering microfacet lobe loses the energy that would have bounced a second
    time between microfacets; the loss grows with roughness and is total nonsense by
    roughness 1. The correction scales the lobe by

        1 + [F_avg * E_avg / (1 - F_avg * (1 - E_avg))] * (1 - E(roughness, mu)) / E(roughness, mu)

    where E is the directional albedo of the SAME lobe with the Fresnel term set to 1 and
    E_avg is that albedo averaged over incidence. For a white conductor (F_avg = 1) the
    bracket is 1 and the scale is exactly 1/E, so a white furnace reads exactly 1 -- which
    is the analytic control this is checked against. For a COLOURED one the bracket is the
    geometric series over repeated tinted bounces; see the comment at the conductor branch
    for why the more commonly quoted one-liner is wrong there and by how much.

    The tables this needs were already computed and SHIPPED by mitsuba3#1822
    (`ggx_E`, `ggx_Eavg`, `ggx_glass_E`, `ggx_glass_inv_E`, and the two glass averages) and
    were then referenced by nothing: `ggx_gen_schlick_ior_s` is the only table any code path
    fetched. Measured consequence before this function existed: a white metallic=1 at
    roughness 0.5 rendered 0.889358 in a furnace where Cycles renders 1.000387, and it was
    BIT-IDENTICAL to Mitsuba's own single-scattering `roughconductor` at the same alpha.

    The glass form has no separate F_avg: the dielectric Fresnel is already inside
    `ggx_glass_E`, whose value is the lobe's TOTAL albedo over reflection and refraction, so
    the scale is 1/E. It reads the SAME table on both sides of the interface, which is a
    compromise forced by a measured defect in the shipped `ggx_glass_inv_E` -- the numbers
    and the probe that established it are at the glass branch below.
    """
    mu = dr.clip(dr.abs(cos_theta), 1e-4, 1.0)
    r = dr.clip(roughness, 1e-4, 1.0)
    if glass:
        assert eta is not None
        # Same z reparameterization the generalized-Schlick table is read with:
        # r0 = ((eta-1)/(eta+1))^2 = z^4.
        eta_up = dr.select(eta >= 1.0, eta, dr.rcp(eta))
        z = dr.sqrt(dr.abs((eta_up - 1.0) / (eta_up + 1.0)))
        E_out = fetch_table("ggx_glass_E").eval([r, mu, z])[0]
        E_in = fetch_table("ggx_glass_inv_E").eval([r, mu, z])[0]
        # DELIBERATELY THE FRONT-SIDE TABLE ON BOTH SIDES, and the reason is a defect in the
        # SHIPPED `ggx_glass_inv_E` table rather than in the choice of formula.
        #
        # The side-correct choice is `ggx_glass_inv_E` for a ray leaving the medium. Applying
        # 1/E per interface with it compounds across the internal bounces and MANUFACTURES
        # energy: white glass at roughness 1.0 goes to 1.426853 where the analytic answer is
        # 1.0 and Cycles reads 0.985881. That looked like the formula compounding, so the
        # DENOMINATOR was measured instead of argued about -- integrate the lobe's own
        # directional albedo straight off `BSDF.sample` (the returned weight IS value/pdf, so
        # its mean is E(mu) by definition) and compare it to the table at the same (r, mu, z).
        # At eta = 1.45, roughness 0.9, a ray LEAVING the medium:
        #
        #     mu     measured E    ggx_glass_inv_E
        #     0.9      0.769731           0.534163      (table 31% low)
        #     0.5      0.639872           0.547019
        #     0.2      0.560146           0.520391
        #
        # The FRONT table passes the same check (0.932259 measured vs 0.908106 tabulated at
        # mu = 0.9), and so does `ggx_E` for the conductor branch, so this is not the probe.
        # `ggx_glass_inv_E` is simply too low, which is exactly how 1/E manufactures energy.
        # Regenerating it means re-deriving mitsuba3's own `precompute.py` -- out of scope
        # here, and it would change a shipped .npy.
        #
        # So of the three reachable options, at white glass roughness 1.0: side-correct
        # tables 1.426853 (manufactures energy), entering interface only 0.626018 (under),
        # E_out on both sides 0.786004 -- the last is monotone in roughness and never exceeds
        # 1, which is the property worth keeping when the alternative is a light source.
        # The residual against Cycles is real and recorded rather than hidden: at
        # transmission = 1 this arm reads -15.9991% at roughness 0.7 and -18.7526% at 0.9,
        # against -27.0303% / -41.5008% uncompensated.
        _ = E_in
        E = E_out
        scale = mi.Spectrum(dr.rcp(dr.maximum(E, 1e-2)))
    else:
        E = dr.maximum(fetch_table("ggx_E").eval([r, mu])[0], 1e-2)
        E_avg = dr.maximum(fetch_table("ggx_Eavg").eval([r])[0], 1e-2)
        f_avg = dr.clip(mi.Spectrum(1.0) if r0 is None else mi.Spectrum(r0), 0.0, 1.0)
        # KULLA-CONTY, TINTED -- and the tint is the whole reason this is not the one-liner
        # `1 + F_avg * (1/E - 1)` that Filament and most engine write-ups give. That form is
        # exact only for a WHITE conductor. For a coloured one it pays out the ENTIRE missing
        # energy at a single bounce's tint, when the light that went missing bounced two,
        # three, n times between microfacets and picked up the tint EACH time. The geometric
        # series over those bounces is Kulla and Conty's
        #
        #     F_ms = F_avg^2 * E_avg / (1 - F_avg * (1 - E_avg))
        #
        # so the multiplier on a single-scattering lobe of albedo F_avg*E is
        # 1 + (F_ms / F_avg) * (1 - E) / E. `k` below is that F_ms / F_avg, written in the
        # reduced form because F_avg cancels -- which is also what keeps a BLACK metal
        # (F_avg = 0) from evaluating 0/0.
        #
        # This is measured, not argued. Grey-0.5 metal at roughness 0.9 in the white furnace:
        # Cycles 0.329946, stock 0.229166 (-30.5444%), the un-tinted one-liner 0.367617
        # (+11.4173% -- it MANUFACTURES energy, which is a worse failure than the deficit it
        # replaces even though the magnitude is smaller). White metal is unchanged by this
        # change: at F_avg = 1 the expression reduces to exactly 1/E, which is the identity
        # the analytic furnace control tests.
        k = f_avg * E_avg / (1.0 - f_avg * (1.0 - E_avg))
        scale = 1.0 + k * (1.0 - E) / E
    # A compensation term is a correction, not a light source: cap it so a table edge or a
    # grazing ray cannot manufacture energy.
    return dr.clip(scale, 1.0, 8.0)


# ------------------------------------------------------------------------------
# Lobes
# ------------------------------------------------------------------------------


class LambertianLobe:
    """
    Lambertian diffuse lobe
    """

    def eval_pdf(wo: mi.Vector3f) -> Tuple[mi.Float, mi.Float]:
        value = dr.abs(mi.Frame3f.cos_theta(wo)) * dr.inv_pi
        return value, value

    def sample(sample2: mi.Point2f) -> mi.Vector3f:
        return mi.warp.square_to_cosine_hemisphere(sample2)


# ------------------------------------------------------------------------------


class OrenNayarLobe:
    """
    Oren Nayar lobe
    """

    def eval_pdf(
        wi: mi.Vector3f,
        wo: mi.Vector3f,
        albedo: mi.UnpolarizedSpectrum,
        roughness: mi.Float,
    ) -> Tuple[mi.Float, mi.Float]:
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        sigma = dr.clip(roughness, 0.0, 1.0)
        A = 1.0 / (dr.pi + sigma * (dr.pi / 2.0 - 2.0 / 3.0))
        B = sigma * A

        # Compute energy compensation term (except for (1.0 - E_o) factor since it depends on wo).
        E_avg = A * dr.pi + ((2 * dr.pi - 5.6) / 3.0) * B
        Ems = (
            dr.inv_pi
            * dr.square(albedo)
            * (E_avg / (1.0 - E_avg))
            / (1.0 - albedo * (1.0 - E_avg))
        )

        def oren_nayar_G(cos_theta):
            sin_theta = dr.sqrt(1.0 - dr.square(cos_theta))
            theta = dr.acos(cos_theta)
            G = sin_theta * (theta - 2.0 / 3.0 - sin_theta * cos_theta) + 2.0 / 3.0 * (
                sin_theta / cos_theta
            ) * (1.0 - dr.square(sin_theta) * sin_theta)
            # The tan(theta) term starts to act up at low cos_theta, so fall back to Taylor expansion.
            G[cos_theta < 1e-6] = (dr.pi / 2.0 - 2.0 / 3.0) - cos_theta
            return G

        E_i = A * dr.pi + B * oren_nayar_G(cos_theta_i)
        multiscatter_term = Ems * (1.0 - E_i)

        t = dr.dot(wi, wo) - cos_theta_o * cos_theta_i
        t[t > 0.0] /= dr.maximum(cos_theta_o, cos_theta_i) + 1e-8

        single_scatter = A + B * t

        E_o = A * dr.pi + B * oren_nayar_G(cos_theta_o)
        multi_scatter = multiscatter_term * (1.0 - E_o)

        value = cos_theta_o * (mi.UnpolarizedSpectrum(single_scatter) + multi_scatter)

        # Fallback to Lambertian when roughness == 0.0
        value[B <= 0.0] = mi.UnpolarizedSpectrum(cos_theta_o * dr.inv_pi)

        pdf = dr.abs(cos_theta_o) * dr.inv_pi

        return value, pdf

    def sample(sample2: mi.Point2f):
        return mi.warp.square_to_cosine_hemisphere(sample2)


# ------------------------------------------------------------------------------


class DiffuseRetroReflectionLobe:
    """
    Diffuse retro reflection diffuse lobe
    """

    def eval_pdf(
        wi: mi.Vector3f, wo: mi.Vector3f, eta: mi.Float, roughness: mi.Float
    ) -> Tuple[mi.Float, mi.Float]:
        """
        Diffuse and retro reflection model
        """
        wh = dr.normalize(wi + wo)
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        F_o = schlick_weight(dr.abs(cos_theta_o))
        F_i = schlick_weight(dr.abs(cos_theta_i))

        # Diffuse reflection
        diff = (1.0 - 0.5 * F_i) * (1.0 - 0.5 * F_o)

        # Retro reflection
        cos_theta_d = dr.dot(wh, wo)
        R_r = 2.0 * roughness * dr.square(cos_theta_d)
        retro = R_r * (F_o + F_i + F_o * F_i * (R_r - 1.0))

        # Combined diffuse and retro reflection
        value = dr.abs(cos_theta_o) * dr.inv_pi * (diff + retro)

        return value, dr.abs(cos_theta_o) * dr.inv_pi

    def sample(sample2: mi.Point2f):
        return mi.warp.square_to_cosine_hemisphere(sample2)


# ------------------------------------------------------------------------------


class FakeSubsurfaceLobe:
    """
    Fake subsurface lobe based on Hanrahan Krueger
    """

    def eval(wi: mi.Vector3f, wo: mi.Vector3f, roughness: mi.Float) -> mi.Float:
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        # Half vector
        wh = dr.normalize(wi + wo)

        F_o = schlick_weight(dr.abs(cos_theta_o))
        F_i = schlick_weight(dr.abs(cos_theta_i))

        # Retro reflection
        cos_theta_d = dr.dot(wh, wo)
        R_r = 2.0 * roughness * dr.square(cos_theta_d)

        # F_ss_90 used to "flatten" retro reflection based on roughness.
        F_ss_90 = R_r / 2.0

        F_ss = dr.lerp(1.0, F_ss_90, F_o) * dr.lerp(1.0, F_ss_90, F_i)
        sss = 1.25 * (
            F_ss * (1.0 / (dr.abs(cos_theta_o) + dr.abs(cos_theta_i)) - 0.5) + 0.5
        )

        # Combined diffuse and retro reflection
        value = dr.abs(cos_theta_o) * dr.inv_pi * sss

        return value


# ------------------------------------------------------------------------------


class SheenLobe:
    """
    Sheen reflection lobe
    """

    def eval(
        wi: mi.Vector3f,
        wo: mi.Vector3f,
        roughness: mi.Float = 0.001,
        anisotropic: mi.Float = 0.0,
    ) -> mi.Float:
        wh = dr.normalize(wi + wo)

        Fd = schlick_weight(dr.abs(dr.dot(wh, wo)))
        value = Fd * dr.abs(mi.Frame3f.cos_theta(wo))

        return value


# ------------------------------------------------------------------------------


class MicrofacetLobe:
    """
    Microfacet diffuse lobe
    """

    def eval_pdf(
        wi: mi.Vector3f,
        wo: mi.Vector3f,
        reflection: bool,
        roughness: mi.Float,
        anisotropic: mi.Float,
        fresnel_mode: MicrofacetFresnel,
        r0: mi.Float = None,
        eta: mi.Float = None,
        m_type=mi.MicrofacetType.GGX,
        correlated_shadow_masking=True,
        has_birefringence: bool = False,
        fast_axis: mi.Float = None,
        retardance: mi.Float = None,
        min_alpha: mi.Float = 0.0,
    ) -> Tuple[mi.Spectrum, mi.Float]:
        # Compute microfacet parameters and instantiate microfacet model
        ax, ay = microfacet_compute_alphas(roughness, anisotropic, min_alpha)
        distr = mi.MicrofacetDistribution(m_type, ax, ay)

        # Eta w.r.t. path
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        front_side = cos_theta_i > 0.0

        if eta is None:
            eta = r0_to_eta(r0)

        eta_path = dr.select(front_side, eta, dr.rcp(eta))

        if r0 is None:
            r0 = eta_to_r0(eta_path)

        wh = half_vector(wi, wo, eta)

        # Shadow masking term
        G, Gi = microfacet_shadow_masking(distr, wi, wo, wh, correlated_shadow_masking)

        if reflection:
            # Evaluate the microfacet distribution (ensure that the half vector points outwards the object)
            D = distr.eval(wh)

            # Evaluate the fresnel term
            F = microfacet_fresnel(fresnel_mode, dr.dot(wi, wh), r0, eta_path)

            common = D * dr.rcp(4.0 * dr.abs(mi.Frame3f.cos_theta(wi)))
            value = mi.Spectrum(common) * F * G
            pdf = mi.Float(common * Gi)
        else:
            # Evaluate the microfacet distribution (ensure that the half vector points outwards the object)
            D = distr.eval(wh)

            dot_wi_wh = dr.dot(wi, wh)
            dot_wo_wh = dr.dot(wo, wh)

            # Evaluate the fresnel term
            F = microfacet_fresnel(fresnel_mode, dot_wi_wh, r0, eta_path)
            T = 1 - F

            # Account for the solid angle compression (not implemented in Cycles)
            scale = 1.0  # dr.square(dr.rcp(eta_path))

            common = dr.abs(
                (dr.square(eta_path) * dot_wo_wh)
                / dr.square(dot_wi_wh + eta_path * dot_wo_wh)
            )
            value = mi.Spectrum(T) * (
                D * G * scale * common * dr.abs(dot_wi_wh / cos_theta_i)
            )
            pdf = distr.pdf(dr.mulsign(wi, cos_theta_i), wh) * common

            valid_F = dr.mean(mi.UnpolarizedSpectrum(F)) >= 1.0
            value[valid_F] = 0.0
            pdf[valid_F] = 0.0
        return value, pdf

    def sample_wh(
        wi: mi.Vector3f,
        sample2: mi.Point2f,
        roughness: mi.Float,
        anisotropic: mi.Float = 0.0,
        m_type=mi.MicrofacetType.GGX,
        min_alpha: mi.Float = 0.0,
    ) -> mi.Vector3f:
        """
        Sample half-vector direction from microfacet model
        """
        ax, ay = microfacet_compute_alphas(roughness, anisotropic, min_alpha)
        distr = mi.MicrofacetDistribution(m_type, ax, ay)
        return distr.sample(dr.mulsign(wi, mi.Frame3f.cos_theta(wi)), sample2)[0]


# ------------------------------------------------------------------------------


class GTR1MicrofacetLobe:
    """
    GTR1 microfacet diffuse lobe
    """

    def eval_pdf(
        wi: mi.Vector3f, wo: mi.Vector3f, roughness: mi.Float, eta: mi.Float = 1.0
    ) -> Tuple[mi.Float, mi.Float]:
        # Half vector
        wh = dr.normalize(wi + wo)
        dot_wi_wh = dr.dot(wi, wh)

        # Evaluate GTR1 isotropic microfacet distribution
        cos_theta_h = mi.Frame3f.cos_theta(wh)
        roughness2 = dr.square(roughness)
        D = (roughness2 - 1.0) / (
            dr.pi
            * dr.log(roughness2)
            * (1.0 + (roughness2 - 1.0) * dr.square(cos_theta_h))
        )
        D = dr.select(D * cos_theta_h > 1e-20, D, 0.0)

        # Shadowing shadowing-masking term
        G = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, 0.25).G(wi, wo, wh)

        # GTR1 uses the schlick approximation for Fresnel term.
        F = fresnel_schlick_approx(dot_wi_wh, 0.04, eta)

        value = mi.Spectrum(F * D * G * mi.Frame3f.cos_theta(wi))
        pdf = mi.Float(cos_theta_h * D * dr.abs(dr.rcp(4.0 * dr.dot(wo, wh))))

        return value, pdf

    def sample_wh(
        wi: mi.Vector3f, sample2: mi.Point2f, roughness: mi.Float
    ) -> mi.Vector3f:
        """
        Sample half-vector direction from microfacet model
        """
        sin_phi, cos_phi = dr.sincos((2.0 * dr.pi) * sample2.x)
        alpha2 = dr.square(roughness)
        cos_theta2 = (1.0 - dr.power(alpha2, 1.0 - sample2.y)) / (1.0 - alpha2)
        sin_theta = dr.sqrt(dr.maximum(0.0, 1.0 - cos_theta2))
        cos_theta = dr.sqrt(dr.maximum(0.0, cos_theta2))
        return mi.Vector3f(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta)
