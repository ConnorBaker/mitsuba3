#include <mitsuba/core/rfilter.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/fwd.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _rfilter-blackman_harris:

Blackman-Harris filter (:monosp:`blackman_harris`)
--------------------------------------------------

.. pluginparameters::

 * - radius
   - |float|
   - Specifies the radius of the filter in pixels (Default: 1.5)

The four-term Blackman-Harris window, applied as a pixel reconstruction filter.
It is strictly non-negative -- so it never rings -- and falls to essentially zero
at its support boundary, which makes it noticeably softer than a tent of the same
radius while remaining much sharper than a Gaussian truncated at the same width.

This filter exists here because it is the DEFAULT pixel filter in Cycles, so it is
what a scene imported from Blender must use if the two renderers are to reconstruct
the same image from the same samples. Its ``radius`` corresponds exactly to Blender's
``scene.cycles.filter_width`` (default 1.5): Cycles doubles that value to obtain the
window's full width and then samples the filter over [0, width/2], so the supported
half-width is the unmodified ``filter_width``.

Writing :math:`u = x / r` for :math:`|u| \le 1`, the filter evaluates to

.. math::
    f(u) = 0.35875 + 0.48829 \cos(\pi u) + 0.14128 \cos(2 \pi u) + 0.01168 \cos(3 \pi u)

which is Cycles' own expression -- it forms :math:`v = 2\pi(x/w + 1/2)` with
:math:`w = 2r` and alternates the signs -- rearranged through
:math:`\cos(\pi(u+1)) = -\cos(\pi u)`. The two agree identically; this form is used
because it makes the peak :math:`f(0) = 1` visible by inspection.

.. tabs::
    .. code-tab:: xml
        :name: blackman_harris-rfilter

        <rfilter type="blackman_harris">
            <float name="radius" value="1.5"/>
        </rfilter>

    .. code-tab:: python

        'type': 'blackman_harris',
        'radius': 1.5,

 */

template <typename Float, typename Spectrum>
class BlackmanHarrisFilter final : public ReconstructionFilter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(ReconstructionFilter, init_discretization, m_radius)
    MI_IMPORT_TYPES()

    BlackmanHarrisFilter(const Properties &props) : Base(props) {
        // Cycles' default `filter_width`, so an unconfigured filter here is an
        // unconfigured filter there.
        m_radius = props.get<ScalarFloat>("radius", 1.5f);
        if (m_radius <= 0.f)
            Throw("BlackmanHarrisFilter: radius must be positive, got %f", m_radius);
        m_inv_radius = 1.f / m_radius;
        init_discretization();
    }

    Float eval(Float x, dr::mask_t<Float> /* active */) const override {
        Float u = x * m_inv_radius,
              t = dr::Pi<Float> * u;

        // The window is only defined on [-1, 1]; outside it the cosines keep
        // oscillating and would hand back spurious positive lobes.
        Float w = 0.35875f + 0.48829f * dr::cos(t)
                           + 0.14128f * dr::cos(2.f * t)
                           + 0.01168f * dr::cos(3.f * t);

        return dr::select(dr::abs(u) <= 1.f, dr::maximum(0.f, w), 0.f);
    }

    std::string to_string() const override {
        return tfm::format("BlackmanHarrisFilter[radius=%f]", m_radius);
    }

    MI_DECLARE_CLASS(BlackmanHarrisFilter)
private:
    ScalarFloat m_inv_radius;
};

MI_EXPORT_PLUGIN(BlackmanHarrisFilter)
NAMESPACE_END(mitsuba)
