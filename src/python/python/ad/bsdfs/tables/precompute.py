# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This code is derived from Blender Cycles and was developed at Meta.

"""
Precomputed GGX tables utilities and CLI.

This module provides:
- fetch_table(name): Load precomputed microfacet-related tables (stored as .npy files
  alongside this module) into Dr.Jit/Mitsuba textures for fast lookup during rendering.
- A command-line interface (when run as a script) to generate and cache these tables.
"""

from __future__ import annotations  # Delayed parsing of type annotations

from os.path import dirname, join

import drjit as dr
import mitsuba as mi
import numpy as np

__tables = {}


def fetch_table(name: str) -> mi.TensorXf:
    """
    Load a precomputed table by name from disk (or cached) as a Dr.Jit texture
    """
    if name not in __tables:
        mi.Log(mi.LogLevel.Debug, f"Load precomputed table: {name}")
        filename = join(dirname(__file__), f"{name}.npy")
        data = mi.TensorXf(np.load(filename))[..., None]
        dr.eval(data)
        texture_type = [mi.Texture1f, mi.Texture2f, mi.Texture3f][len(data.shape) - 2]
        __tables[name] = texture_type(data)
    return __tables[name]


# ------------------------------------------------------------------------------
# Table pre-computation
# ------------------------------------------------------------------------------
#
# HSR: THIS GENERATOR COULD NOT REPRODUCE A SINGLE ONE OF THE TABLES IT SHIPS, AND IT
# COULD NOT RUN AT ALL. Everything below `fetch_table` used to sit inside
# `if __name__ == "__main__":`, so nothing imported it, nothing called it, and no test
# could reach it. Run as documented (`python -m mitsuba.ad.bsdfs.tables.precompute`) it
# stopped on the first line, and each repair exposed the next:
#
#   1. `mi.set_variant("llvm_ad_rgb")` -- hard-coded, and this build enables
#      `scalar_rgb` / `cuda_ad_rgb` only.  ImportError.
#   2. `from .lobes import ...` -- this module is `ad.bsdfs.tables.precompute`, so the
#      relative import resolves to `ad.bsdfs.tables.lobes`, which does not exist.
#   3. `sanitize_eta` is in `ad.bsdfs.common`, not in `ad.bsdfs.lobes`.  ImportError.
#   4. `compute_table` did `weight[~dr.isfinite(data)] = 0.0` with `data` a Spectrum for
#      the last table, which is a Float indexed by an Array3f mask.  TypeError, after
#      six wrong tables had already been written to disk.
#
# With those four cleared it ran to completion and OVERWROTE the shipped tables with
# values that are wrong by up to 0.68 absolute (`ggx_glass_inv_E`), 2410 of 4096 cells
# beyond 0.01 on `ggx_glass_E`, 674 of 1024 on `ggx_E`.  Two estimator defects explain
# it, and both are fixed below:
#
#   A. NO BELOW-HORIZON REJECTION.  `microfacet_shadow_masking` calls
#      `distr.smith_g1(w, [0, 0, 1])`, whose second argument exists to defeat Mitsuba's
#      "consistent orientation" test -- but that test is also what makes `G` vanish for
#      an outgoing direction under the surface, so the masking term happily shadows a
#      direction that is not there.  The lobe's CALLERS mask it (`_eval_pdf_impl` gates
#      every lobe on `reflect_shading` / `refract_shading`), so this never reaches a
#      render; the table generator had no such mask and double-counted.  Adding the
#      rejection takes `ggx_E` from mean |d| 5.198e-02 against the shipped table to
#      3.058e-05, and `ggx_Eavg` from 6.695e-02 to 1.563e-06 -- i.e. the shipped tables
#      are right and this estimator was wrong.
#   B. A SPURIOUS `* dr.square(eta)` ON THE REFRACTION TERM.  It compensates the
#      solid-angle compression factor, which `MicrofacetLobe.eval_pdf` does not apply --
#      `scale = 1.0  # dr.square(dr.rcp(eta_path))` -- so the product was the RADIANCE
#      albedo where 1/E needs the ENERGY one.  Removing it takes `ggx_glass_inv_E` from
#      7.710e-02 to 1.249e-03 and `ggx_glass_inv_Eavg` from 1.063e-01 to 8.773e-04.
#
# The corrected estimator is checked against an INDEPENDENT oracle rather than against
# itself: Mitsuba's C++ `roughdielectric` sampled in `TransportMode::Importance` (whose
# mean sample weight is the energy albedo by construction) agrees with it to four digits
# at eta 1.38 / 1.45 / 3.42, and a brute-force hemisphere integral of `D*G/(4 cos_i)`
# agrees with the shipped `ggx_E` to 2e-4.  `src/python_tests/test_ggx_table_provenance.py`
# pins all of it.
#
# STILL OPEN, recorded rather than fixed: `ggx_glass_E` / `ggx_glass_Eavg` reproduce
# everywhere EXCEPT the top of the `z` axis (`z -> 1`, i.e. eta -> 3.42, the clip of
# `reparam_eta`), where the shipped values collapse onto the CONDUCTOR albedo
# (`ggx_glass_E[15,15,15] = 0.323696` against `ggx_E`'s `0.307076`) while both the
# corrected estimator and `roughdielectric` read `0.697`.  Whatever produced that corner
# is not this code.  Real glass sits at eta 1.45, i.e. `z ~ 0.43`, so nothing in the
# Blender parity work reads it; the test asserts the disagreement stays confined there
# rather than skipping the table.


_DEPS = None


def _deps():
    """Deferred import of the lobe routines.

    `ad.bsdfs.lobes` imports `fetch_table` from THIS module, so a module-level import
    here would close the cycle. It is deferred rather than kept inside `__main__`
    because a generator no test can call is a generator nothing checks -- which is how
    the four import defects above survived.

    ABSOLUTE, not relative. `from .lobes import ...` was the second defect (this module
    lives in the `tables` sub-package, so it resolved to a module that does not exist),
    and an absolute name also lets a test load this file straight from the checkout to
    ask whether the CHECKED-IN generator reproduces the CHECKED-IN tables -- a question
    about the source tree, which the built package cannot answer.
    """
    global _DEPS
    if _DEPS is None:
        from mitsuba.ad.bsdfs.common import sanitize_eta
        from mitsuba.ad.bsdfs.lobes import (
            MicrofacetFresnel,
            MicrofacetLobe,
            microfacet_reflect,
            microfacet_refract,
            r0_to_eta,
        )

        _DEPS = {
            "MicrofacetFresnel": MicrofacetFresnel,
            "MicrofacetLobe": MicrofacetLobe,
            "microfacet_reflect": microfacet_reflect,
            "microfacet_refract": microfacet_refract,
            "r0_to_eta": r0_to_eta,
            "sanitize_eta": sanitize_eta,
        }
    return _DEPS


def compute_table(name: str, samples: int, dims: list, func, write: bool = True):
    """
    Table pre-computation routine. The function to evaluate should take a number
    of arguments equal to the number of dimensions (==len(dims)) plus one for the
    3D random points.

    Parameters:
        name:    Name of the table
        samples: Number of samples to use for Monte Carlo integration
        dims:    Dimensions of the table
        func:    Function to evaluate
        write:   Write the result next to this module. Off for tests, which must be
                 able to regenerate a table without replacing the shipped one.

    Returns the table as a NumPy array.
    """
    dims = list(reversed(dims))

    mi.Log(mi.LogLevel.Info, f"{name}: table computation")

    # Generate the axis arrays
    axis = dr.meshgrid(
        *[dr.linspace(mi.Float, 1e-4, 1.0, d, endpoint=True) for d in dims],
        dr.arange(mi.Float, samples),
        indexing="ij",
    )
    seed = axis[-1]
    axis = list(reversed(axis[:-1]))

    # Generate 3D random numbers
    v0, v1 = mi.sample_tea_32(dr.arange(mi.UInt32, dr.width(seed)), seed)
    rng = mi.PCG32(initstate=v0, initseq=v1)
    sample2 = mi.Point2f(rng.next_float32(), rng.next_float32())
    rand = rng.next_float32()

    # Evaluate the function
    data, weight = func(*axis, sample2, rand)

    # Sanitize output values. HSR: the mask is taken ONCE, before `data` is repaired --
    # it used to be recomputed for the weight line AFTER `data` had already been made
    # finite, so that line was a no-op and a non-finite sample was dropped from the
    # numerator while still counting in the denominator.
    bad = ~dr.isfinite(data)
    data[bad] = 0.0
    weight[bad] = 0.0

    # Accumulate sample values in final buffer
    cell_id = dr.arange(mi.UInt32, dr.width(seed)) // samples

    table = dr.zeros(mi.TensorXf, shape=dims)
    table_weight = dr.zeros(mi.TensorXf, shape=dims)
    dr.scatter_reduce(dr.ReduceOp.Add, table.array, data, cell_id)
    dr.scatter_reduce(dr.ReduceOp.Add, table_weight.array, weight, cell_id)

    # Sanitize output table and weights
    table.array[~dr.isfinite(table.array)] = 0.0
    table_weight.array[table_weight.array == 0.0] = 1.0

    # Normalize samples
    table /= table_weight
    table = dr.clip(table, 0.0, 1.0)

    result = np.array(table)

    if write:
        filename = join(dirname(__file__), f"{name}.npy")
        mi.Log(mi.LogLevel.Info, f"  -> writing to file: {filename}")
        np.save(filename, result)

    return result


def ggx_E(
    roughness,
    mu,
    sample2,
    fresnel_mode,
    eta=1.0,
    r0=None,
    reject_below_horizon: bool = True,
    solid_angle_compression: bool = False,
):
    """
    GGX sampling and evaluation routine used in pre-computations.

    Returns `(value, weight)`: the per-sample estimate of the lobe's DIRECTIONAL ALBEDO
    -- the fraction of incident energy the single-scattering lobe delivers -- and 1 for
    every sample that was drawn.

    The two boolean arguments exist so a test can reintroduce each historical defect and
    watch the check fail; neither should be touched by a caller that wants the tables.

        reject_below_horizon    Discard the CONTRIBUTION (never the sample) of an
                                outgoing direction on the wrong side of the surface.
                                `False` reproduces defect A above.
        solid_angle_compression `True` multiplies the refracted term by eta^2, which is
                                correct only if `MicrofacetLobe.eval_pdf` applies the
                                1/eta^2 radiance factor. It does not. Reproduces defect B.
    """
    d = _deps()
    MicrofacetLobe = d["MicrofacetLobe"]

    result = mi.Float(0.0)
    valid = mi.Bool(False)

    wi = mi.Vector3f(dr.sqrt(1.0 - dr.square(mu)), 0.0, mu)
    wh = MicrofacetLobe.sample_wh(wi, sample2, roughness, 0.0)

    wo_reflect = d["microfacet_reflect"](wi, wh)
    reflect_value, reflect_pdf = MicrofacetLobe.eval_pdf(
        wi,
        wo=wo_reflect,
        reflection=True,
        roughness=roughness,
        anisotropic=0.0,
        fresnel_mode=fresnel_mode,
        r0=r0,
        eta=eta,
    )
    valid_reflect = reflect_pdf != 0.0
    carries = valid_reflect
    if reject_below_horizon:
        carries = carries & (mi.Frame3f.cos_theta(wo_reflect) > 0.0)
    result += dr.select(carries, mi.luminance(reflect_value) / reflect_pdf, 0.0)
    valid |= valid_reflect

    if fresnel_mode == d["MicrofacetFresnel"].DIELECTRIC:
        wo_refract = d["microfacet_refract"](wi, wh, eta)
        refract_value, refract_pdf = MicrofacetLobe.eval_pdf(
            wi,
            wo=wo_refract,
            reflection=False,
            roughness=roughness,
            anisotropic=0.0,
            fresnel_mode=fresnel_mode,
            r0=r0,
            eta=eta,
        )
        valid_refract = refract_pdf != 0.0
        carries = valid_refract
        if reject_below_horizon:
            carries = carries & (mi.Frame3f.cos_theta(wo_refract) < 0.0)
        scale = dr.square(eta) if solid_angle_compression else mi.Float(1.0)
        result += dr.select(
            carries, mi.luminance(refract_value) / refract_pdf * scale, 0.0
        )
        valid |= valid_refract

    return result, dr.select(valid, 1.0, 0.0)


def ggx_gen_schlick_s(roughness, mu, sample2, fresnel_mode, eta,
                      reject_below_horizon: bool = True):
    """
    Interpolation factor between F0 and F90 for the generalized Schlick Fresnel.

    Returns a FLOAT, not a Spectrum. The ratio is achromatic by construction (the two
    evaluations differ only in `r0`, a scalar), and `compute_table` accumulates into a
    scalar cell -- handing it a Spectrum is what made the shipped script raise
    `TypeError` before it could write this table.
    """
    d = _deps()
    MicrofacetLobe = d["MicrofacetLobe"]

    wi = mi.Vector3f(dr.sqrt(1.0 - dr.square(mu)), 0.0, mu)
    wh = MicrofacetLobe.sample_wh(wi, sample2, roughness, 0.0)
    wo = d["microfacet_reflect"](wi, wh)

    value_0, pdf = MicrofacetLobe.eval_pdf(
        wi, wo, reflection=True, roughness=roughness, anisotropic=0.0,
        fresnel_mode=fresnel_mode, eta=eta, r0=0.0,
    )
    value_1 = MicrofacetLobe.eval_pdf(
        wi, wo, reflection=True, roughness=roughness, anisotropic=0.0,
        fresnel_mode=fresnel_mode, eta=eta, r0=1.0,
    )[0]

    valid = pdf != 0.0
    if reject_below_horizon:
        valid = valid & (mi.Frame3f.cos_theta(wo) > 0.0)
    return mi.luminance(value_0) / mi.luminance(value_1), dr.select(valid, 1.0, 0.0)


def reparam_eta(z):
    """
    Parameterization ensuring the entire [1..inf] range of IORs is covered
    """
    d = _deps()
    z = dr.clip(z, 1e-4, 0.99)
    return d["sanitize_eta"](d["r0_to_eta"](dr.square(dr.square(z))))


def table_specs(**kwargs):
    """
    The seven shipped tables, as `(name, samples, dims, func)`.

    `kwargs` are forwarded to the estimators, which is how a test reintroduces a defect.
    """
    F = _deps()["MicrofacetFresnel"]

    def albedo(r, mu, s2, rand):
        return ggx_E(r, mu, s2, F.NONE, **kwargs)

    def albedo_avg(r, s2, rand):
        value, weight = ggx_E(r, rand, s2, F.NONE, **kwargs)
        return 2.0 * rand * value, weight

    def glass(r, mu, z, s2, rand):
        return ggx_E(r, mu, s2, F.DIELECTRIC, eta=reparam_eta(z), **kwargs)

    def glass_avg(r, z, s2, rand):
        value, weight = ggx_E(r, rand, s2, F.DIELECTRIC, eta=reparam_eta(z), **kwargs)
        return 2.0 * rand * value, weight

    def glass_inv(r, mu, z, s2, rand):
        return ggx_E(r, mu, s2, F.DIELECTRIC, eta=dr.rcp(reparam_eta(z)), **kwargs)

    def glass_inv_avg(r, z, s2, rand):
        value, weight = ggx_E(
            r, rand, s2, F.DIELECTRIC, eta=dr.rcp(reparam_eta(z)), **kwargs
        )
        return 2.0 * rand * value, weight

    def schlick_s(r, mu, z, s2, rand):
        return ggx_gen_schlick_s(
            r, mu, s2, F.GENERALIZED_SCHLICK, reparam_eta(z),
            reject_below_horizon=kwargs.get("reject_below_horizon", True),
        )

    return [
        # Albedo of the GGX microfacet BRDF, roughness X incident direction
        ("ggx_E", int(1e5), [32, 32], albedo),
        # Averaged over incident direction
        ("ggx_Eavg", int(1e5), [32], albedo_avg),
        # Overall albedo of the GGX microfacet BSDF with dielectric Fresnel,
        # roughness X incident direction X IOR, for IOR>1
        ("ggx_glass_E", int(1e3), [16, 16, 16], glass),
        ("ggx_glass_Eavg", int(1e6), [16, 16], glass_avg),
        # ... and for IOR<1
        ("ggx_glass_inv_E", int(1e3), [16, 16, 16], glass_inv),
        ("ggx_glass_inv_Eavg", int(1e5), [16, 16], glass_inv_avg),
        # Interpolation factor between F0 and F90 for the generalized Schlick Fresnel,
        # depending on cosI and roughness, for IOR>1, using dielectric Fresnel mode.
        ("ggx_gen_schlick_ior_s", int(1e4), [16, 16, 16], schlick_s),
    ]


def generate_all(write: bool = True, only=None, **kwargs):
    """Regenerate every shipped table (or the subset named in `only`)."""
    out = {}
    for name, samples, dims, func in table_specs(**kwargs):
        if only is not None and name not in only:
            continue
        out[name] = compute_table(name, samples, dims, func, write=write)
    return out


if __name__ == "__main__":
    # HSR: not hard-coded to `llvm_ad_rgb` -- that variant is not enabled in every build,
    # and it was the first thing that stopped this script from running.
    if mi.variant() is None:
        for _v in ("llvm_ad_rgb", "cuda_ad_rgb", "scalar_rgb"):
            if _v in mi.variants():
                mi.set_variant(_v)
                break
        else:
            raise RuntimeError("no suitable Mitsuba variant is enabled")

    generate_all()
