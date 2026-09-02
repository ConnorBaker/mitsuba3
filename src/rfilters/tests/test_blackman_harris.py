"""The Blackman-Harris filter exists to reconstruct the same image Cycles does, so the
test transcribes CYCLES' expression -- not the plugin's -- and demands they agree.

The plugin evaluates a rearranged form (the signs alternate the other way and the phase
shift is gone), which is algebraically identical but is NOT the same sequence of
operations. Comparing the plugin against its own formula would only prove it can copy;
comparing it against `filter_func_blackman_harris` from `intern/cycles/scene/film.cpp`
is what makes this a parity test.

The width convention is the trap and is checked explicitly. Cycles' table builder does
`width *= 2.0f` for FILTER_BLACKMAN_HARRIS and then samples the filter over
[0, width * 0.5], so the half-width actually rendered equals the UNMODIFIED
`filter_width` -- radius `r` here corresponds to `cycles.filter_width = r`, with no
factor of two anywhere. An exporter that passes `filter_width` straight through as
`radius` is correct; one that doubles or halves it is not, and only this test would say so.
"""

import math

import pytest
import drjit as dr
import mitsuba as mi


def cycles_blackman_harris(v, filter_width):
    """`filter_func_blackman_harris` from intern/cycles/scene/film.cpp, verbatim,
    with the `width *= 2.0f` the table builder applies before calling it."""
    width = filter_width * 2.0
    if abs(v) > width * 0.5:      # the table is only built over [0, width/2]
        return 0.0
    v = 2.0 * math.pi * (v / width + 0.5)
    return (0.35875
            - 0.48829 * math.cos(v)
            + 0.14128 * math.cos(2.0 * v)
            - 0.01168 * math.cos(3.0 * v))


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5, 2.0, 3.0])
def test01_matches_cycles_expression(variants_any_scalar, radius):
    f = mi.load_dict({'type': 'blackman_harris', 'radius': radius})
    assert f.radius() == pytest.approx(radius)
    # Sample well past the support so the out-of-window behaviour is covered too.
    for i in range(201):
        x = -2.0 * radius + (4.0 * radius) * i / 200.0
        assert f.eval(x) == pytest.approx(cycles_blackman_harris(x, radius), abs=1e-6), x


def test02_peaks_at_one_and_vanishes_at_the_edge(variants_any_scalar):
    f = mi.load_dict({'type': 'blackman_harris', 'radius': 1.5})
    assert f.eval(0.0) == pytest.approx(1.0, abs=1e-6)
    # 0.35875 - 0.48829 + 0.14128 - 0.01168; the window does not reach exactly zero.
    assert f.eval(1.5) == pytest.approx(6e-5, abs=1e-6)
    assert f.eval(-1.5) == pytest.approx(6e-5, abs=1e-6)


def test03_never_negative(variants_any_scalar):
    """A negative lobe is what makes a filter ring. This window has none, and a
    reconstruction filter that develops one would be a different filter."""
    f = mi.load_dict({'type': 'blackman_harris', 'radius': 1.5})
    for i in range(1001):
        x = -3.0 + 6.0 * i / 1000.0
        assert f.eval(x) >= 0.0, x


def test04_outside_the_support_it_is_zero_and_stays_zero(variants_any_scalar):
    """The cosines keep oscillating past |u| = 1 and would hand back spurious lobes
    if the support were not clamped. This is the failure the `select` prevents."""
    f = mi.load_dict({'type': 'blackman_harris', 'radius': 1.0})
    for x in (1.0001, 1.5, 2.0, 2.5, 3.0, 10.0, -1.0001, -2.0, -10.0):
        assert f.eval(x) == 0.0, x


def test05_it_is_not_the_gaussian_it_would_otherwise_fall_back_to(variants_any_scalar):
    """Guard against the whole point being lost: if `blackman_harris` silently resolved
    to something else, every test above could still pass on a lucky approximation. It
    must differ measurably from the filter a scene gets when none is specified."""
    bh = mi.load_dict({'type': 'blackman_harris', 'radius': 1.5})
    g  = mi.load_dict({'type': 'gaussian'})   # stddev 0.5, radius 2.0 -- Mitsuba's default
    assert bh.radius() != g.radius()
    assert abs(bh.eval(1.0) - g.eval(1.0)) > 0.05


def test06_a_nonpositive_radius_is_refused(variants_any_scalar):
    for r in (0.0, -1.0):
        with pytest.raises(RuntimeError):
            mi.load_dict({'type': 'blackman_harris', 'radius': r})
