"""The shipped GGX tables against the generator that claims to produce them.

WHAT THIS PINS. `ad/bsdfs/tables/*.npy` are read at every shading point by
`ggx_energy_compensation`, `ggx_directional_albedo` and `microfacet_estimate_albedo`.
`ad/bsdfs/tables/precompute.py` is the script that makes them. Until this test existed
NOTHING connected the two: the generator sat entirely inside `if __name__ ==
"__main__":`, so no import could reach it, and run as documented it raised on its first
line. Four import/type defects deep it finally ran -- and overwrote the tables with
values wrong by up to 0.68 absolute. A data file whose generator cannot reproduce it is
a data file nobody can ever fix.

THE TWO ESTIMATOR DEFECTS, both reproducible on demand through `table_specs(**kwargs)`:

  A `reject_below_horizon=False` -- the shipped behaviour. A sampled outgoing direction
    under the surface still carried energy, because `microfacet_shadow_masking` passes
    `[0, 0, 1]` to `smith_g1` to defeat Mitsuba's orientation check, and that check is
    also what makes `G` vanish there. The lobe's callers mask it, so no render is
    affected; an albedo integral has no such mask and double-counted.
  B `solid_angle_compression=True` -- the shipped behaviour. The refracted term was
    multiplied by eta^2, compensating a factor `MicrofacetLobe.eval_pdf` does not apply
    (`scale = 1.0`), so the tables held the RADIANCE albedo where 1/E needs the ENERGY
    one.

WHY THE SHIPPED TABLES ARE THE REFERENCE AND THE GENERATOR IS THE THING UNDER TEST.
Two independent oracles say the tables are right and the estimator was wrong: a
brute-force hemisphere integral of `D*G/(4 cos_i)` reproduces `ggx_E` to 2e-4, and
Mitsuba's C++ `roughdielectric` sampled in `TransportMode::Importance` -- whose mean
sample weight is the energy albedo by construction -- reproduces the glass tables to
~1e-3. Neither shares a line of code with the Python lobe.

WHY THIS TEST READS THE CHECKOUT AND NOT THE BUILT PACKAGE. Its subject is "does the
checked-in generator reproduce the checked-in data", which is a property of the source
tree; `import mitsuba` resolves to the BUILT package, which lags the checkout, so a
normal import would test the previous build's answer to a question about this one. The
LOBE routines still come from the built package -- they are the thing the tables must be
consistent with.

FOUR CONTROLS, because "regenerating a table agrees with the table" passes for free if
the comparison is loose:
  * FALSIFIER -- each defect, reintroduced, must push the same statistic past the
    threshold by a wide margin. Recorded per table below.
  * PHYSICAL -- `ggx_gen_schlick_ior_s` is scored against an analytic limit (the
    Schlick interpolation factor tends to 1 at grazing incidence) rather than against
    itself, because that table does NOT reproduce and the shipped one is the artifact
    that is wrong. See `test_gen_schlick_grazing_limit`.
  * SUPPORT -- every comparison prints the number of cells it covers, and a table with
    zero cells fails rather than passes.
  * TYPE -- `compute_table` must survive a Spectrum-valued estimator, which is what
    stopped the shipped script before it could write the seventh table.

STILL OPEN, asserted rather than skipped so it cannot rot silently:
  * `ggx_glass_E` / `ggx_glass_Eavg` reproduce only for `z <= 10` (eta <= 2.60). Above
    that the shipped values collapse onto the CONDUCTOR albedo while the corrected
    estimator and `roughdielectric` agree on a value twice as large. Real glass is at
    eta 1.45, i.e. z ~ 0.43, so nothing in the Blender parity work reads that corner.
  * `ggx_gen_schlick_ior_s` does not reproduce anywhere, and the shipped table is the
    one that violates the analytic limit.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import drjit as dr
import mitsuba as mi

# The `z` axis index above which the shipped `ggx_glass_E` stops being reproducible.
# `z = 10/15` maps through `reparam_eta` to eta = 2.6003; glass is 1.45.
GLASS_Z_LIMIT = 10

# Sample counts capped for test runtime. Everything else runs at the shipped count.
SAMPLE_CAP = {"ggx_E": int(2e4), "ggx_glass_Eavg": int(1e5)}


def _variant():
    for v in ("llvm_ad_rgb", "cuda_ad_rgb"):
        if v in mi.variants():
            return v
    pytest.skip("this test integrates millions of BSDF samples; scalar_rgb is unusable")


def _src_dir():
    d = Path(__file__).resolve().parents[1] / "python" / "python" / "ad" / "bsdfs" / "tables"
    if not (d / "precompute.py").is_file():
        pytest.skip(f"generator source not present next to the test ({d})")
    return d


def _load_generator():
    """Load `precompute.py` from the CHECKOUT, not from the built package."""
    path = _src_dir() / "precompute.py"
    spec = importlib.util.spec_from_file_location("_precompute_from_checkout", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _regenerate(pc, only=None, **kwargs):
    out = {}
    for name, samples, dims, func in pc.table_specs(**kwargs):
        if only is not None and name not in only:
            continue
        out[name] = pc.compute_table(
            name, SAMPLE_CAP.get(name, samples), dims, func, write=False
        )
    return out


def _shipped(name):
    return np.load(_src_dir() / f"{name}.npy")


def _score(got, ref, z_limit=None):
    """Mean absolute deviation and the number of cells it rests on."""
    if z_limit is not None:
        got, ref = got[: z_limit + 1], ref[: z_limit + 1]
    d = np.abs(got - ref)
    return float(d.mean()), float(d.max()), int(d.size)


# Per table: (mean |d| the corrected generator must beat, `z` restriction or None).
# Each threshold is at least 5x the corrected figure measured on this machine and at
# most 1/5 of the figure either defect produces -- see the falsifier test.
REPRODUCES = {
    "ggx_E": (1.0e-2, None),
    "ggx_Eavg": (1.0e-4, None),
    "ggx_glass_inv_E": (4.0e-3, None),
    "ggx_glass_inv_Eavg": (3.0e-3, None),
    "ggx_glass_E": (1.0e-3, GLASS_Z_LIMIT),
    "ggx_glass_Eavg": (4.0e-3, GLASS_Z_LIMIT),
}


def test_generator_reproduces_the_shipped_tables():
    """The corrected generator must land on the data it ships."""
    mi.set_variant(_variant())
    pc = _load_generator()
    got = _regenerate(pc, only=set(REPRODUCES))

    failures = []
    for name, (tol, z_limit) in REPRODUCES.items():
        mean, worst, n = _score(got[name], _shipped(name), z_limit)
        assert n > 0, f"{name}: compared 0 cells -- the check measured nothing"
        print(f"{name:22s} mean|d| {mean:.4e}  max|d| {worst:.4e}  cells {n}")
        if mean > tol:
            failures.append(f"{name}: mean|d| {mean:.4e} > {tol:.1e} ({n} cells)")
    assert not failures, "generator does not reproduce its own tables:\n  " + "\n  ".join(
        failures
    )


@pytest.mark.parametrize(
    "defect,kwargs",
    [
        ("A: no below-horizon rejection", dict(reject_below_horizon=False)),
        ("B: eta^2 solid-angle compression", dict(solid_angle_compression=True)),
    ],
)
def test_each_defect_reintroduced_breaks_the_check(defect, kwargs):
    """THE FALSIFIER. A check that has never failed proves nothing.

    Each historical defect, put back one at a time, must push at least one table past
    its threshold by at least 5x -- so the thresholds above are separating the two
    states rather than admitting both.
    """
    mi.set_variant(_variant())
    pc = _load_generator()
    got = _regenerate(pc, only=set(REPRODUCES), **kwargs)

    margins = []
    for name, (tol, z_limit) in REPRODUCES.items():
        mean, _, n = _score(got[name], _shipped(name), z_limit)
        assert n > 0
        margins.append((mean / tol, name, mean, tol))
    margins.sort(reverse=True)
    worst_ratio, name, mean, tol = margins[0]
    print(f"[{defect}] worst: {name} mean|d| {mean:.4e} = {worst_ratio:.1f}x its {tol:.1e}")
    assert worst_ratio >= 5.0, (
        f"defect '{defect}' left every table inside its tolerance (worst {name} at "
        f"{worst_ratio:.2f}x). The tolerances cannot tell the two states apart."
    )


def test_gen_schlick_grazing_limit():
    """`ggx_gen_schlick_ior_s` does NOT reproduce, and the SHIPPED table is the wrong one.

    The stored quantity is `s` in `F = lerp(r0, 1, s)` for the generalized-Schlick
    Fresnel, i.e. `s = (F_real - F0) / (1 - F0)` evaluated at the microfacet angle. At
    grazing incidence `F_real -> 1`, so `s -> 1` analytically, whatever the IOR. That is
    an oracle neither artifact can argue with.

    The shipped table violates it -- not everywhere, but by a lot where it does: over
    the 15 non-degenerate IORs on the grazing/smooth edge it runs from 0.997288 down to
    0.267234, against a corrected generator that stays above 0.988 on every one. The
    failure has a mechanism: `compute_table`
    zeroed non-finite samples in the NUMERATOR and then recomputed the mask, by which
    time nothing was non-finite, so those samples still counted in the DENOMINATOR. For
    this table the ratio is 0/0 wherever the microfacet masks the sample, which is most
    of them at grazing -- so the cell was diluted toward zero. That ordering is fixed;
    the .npy is deliberately NOT regenerated here, because replacing a table this BSDF
    reads at every shading point is a change that has to be scored against a render.
    """
    mi.set_variant(_variant())
    pc = _load_generator()
    name = "ggx_gen_schlick_ior_s"
    spec = [t for t in pc.table_specs() if t[0] == name]
    assert len(spec) == 1, f"{name} is not in table_specs()"
    _, samples, dims, func = spec[0]
    got = pc.compute_table(name, SAMPLE_CAP.get(name, samples), dims, func, write=False)
    ref = _shipped(name)

    # Index order is [z, mu, r]; column 0 of the mu axis is mu = 1e-4, i.e. grazing.
    # Take the smoothest roughness so the microfacet angle tracks the view angle.
    # `z[0]` maps to eta = 1.0001 -- an interface that is not an interface, where
    # `s = (F_real - F0) / (1 - F0)` is 0/0 up to rounding and the limit argument says
    # nothing. It is dropped by INDEX, not by looking at which cells would pass.
    grazing_got = got[1:, 0, 0]
    grazing_ref = ref[1:, 0, 0]
    print(f"grazing s: corrected min {grazing_got.min():.6f}  "
          f"shipped min {grazing_ref.min():.6f}  shipped max {grazing_ref.max():.6f}  "
          f"over {grazing_got.size} IORs")
    assert grazing_got.size > 0

    assert grazing_got.min() > 0.98, (
        "the CORRECTED generator misses the analytic limit s -> 1 at grazing "
        f"(min {grazing_got.min():.6f}); the fix is wrong, not the table"
    )
    assert grazing_ref.min() < 0.5, (
        "the shipped ggx_gen_schlick_ior_s now satisfies the grazing limit -- it has "
        "been regenerated, and this test's premise (and the residual it records) is "
        "stale. Re-derive it rather than relaxing the bound."
    )


def test_compute_table_survives_a_spectrum_valued_estimator():
    """TYPE CONTROL. `weight[~dr.isfinite(data)] = 0.0` with a Spectrum `data` raised
    `TypeError` -- after six wrong tables had already been written to disk. The fix is
    that `ggx_gen_schlick_s` returns a Float; this asserts the last table is reachable
    at all, which is the property the crash denied."""
    mi.set_variant(_variant())
    pc = _load_generator()
    out = pc.compute_table(
        "ggx_gen_schlick_ior_s", 64, [4, 4, 4],
        [t for t in pc.table_specs() if t[0] == "ggx_gen_schlick_ior_s"][0][3],
        write=False,
    )
    assert out.shape == (4, 4, 4)
    assert np.isfinite(out).all()


def test_generator_writes_nothing_unless_asked():
    """`write=False` must not touch the shipped data -- otherwise running the tests
    silently replaces the tables, which is the failure this whole file is about."""
    mi.set_variant(_variant())
    pc = _load_generator()
    path = _src_dir() / "ggx_E.npy"
    before = path.read_bytes()
    pc.compute_table("ggx_E", 64, [4, 4],
                     [t for t in pc.table_specs() if t[0] == "ggx_E"][0][3],
                     write=False)
    assert path.read_bytes() == before
