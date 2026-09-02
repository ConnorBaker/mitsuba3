#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-uniform:

Uniform spectrum (:monosp:`uniform`)
------------------------------------

.. pluginparameters::

 * - wavelength_min
   - |float|
   - Lower bound of the wavelength sampling range in nanometers. Default: 360 nm

 * - wavelength_max
   - |float|
   - Upper bound of the wavelength sampling range in nanometers. Default: 830 nm

 * - value
   - |float|
   - Value of the spectral function across the specified spectral range.
   - |exposed|, |differentiable|

This spectrum returns a constant reflectance or emission value over the spectral
dimension. It implements a uniform sampling method on a finite spectral range
controlled by the ``wavelength_min`` and ``wavelength_max`` parameters.

.. tabs::
    .. code-tab:: xml
        :name: uniform

        <spectrum type="uniform">
            <float name="value" value="0.1"/>
        </spectrum>

    .. code-tab:: python

        'type': 'uniform',
        'value': 0.1

 */

template <typename Float, typename Spectrum>
class UniformSpectrum final : public SurfaceField<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(SurfaceField, Texture)

    UniformSpectrum(const Properties &props) : SurfaceField(props) {
        m_value = dr::opaque<Float>(props.get<ScalarFloat>("value"));
        m_range = ScalarVector2f(props.get<ScalarFloat>("wavelength_min", MI_CIE_MIN),
                                 props.get<ScalarFloat>("wavelength_max", MI_CIE_MAX));
    }

    void traverse(TraversalCallback *cb) override {
        cb->put("value", m_value, ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {
        if constexpr (dr::is_jit_v<Float>)
            if (unlikely(m_value.size() != 1))
                Throw("Updated the uniform spectrum with a float of size %d", m_value.size());
        dr::make_opaque(m_value);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f & /*si*/,
                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_spectral_v<Spectrum>)
            return UnpolarizedSpectrum(m_value);
        else
            return m_value;
    }

    Float eval_1(const SurfaceInteraction3f & /*it*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return m_value;
    }

    Color3f eval_3(const SurfaceInteraction3f & /*it*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return Color3f(m_value);
    }

    Vector2f eval_1_grad(const SurfaceInteraction3f & /*it*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return 0.0;
    }

    /* THE DENSITY OF `sample_spectrum()` BELOW, which maps a uniform `sample` affinely
       onto `m_range` -- so the density is the constant `1 / (range.y - range.x)` inside
       the range and 0 outside it.  This USED to be `NotImplementedError("pdf")`, which
       was harmless while nothing asked: `sample_spectrum` returns eval/pdf directly and
       never consults this.  It stopped being harmless when `Sensor::sample_wavelengths`
       began weighting a sensor-level `srf` by the true inverse sampling PDF (this fork's
       ef5383de, porting upstream PR #1710's reading) -- an `srf` of type `uniform`, and a
       bare float, which resolves to this plugin, then raised at render time where the
       previous `sample_spectrum` path had worked.  Implementing the density is the fix
       rather than special-casing the sensor: the value is not a guess or a fallback, it is
       exactly the density of the sampling routine ten lines below. */
    Wavelength pdf_spectrum(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_spectral_v<Spectrum>) {
            /* `active` is a per-LANE mask and the range test is per-WAVELENGTH (4 wide),
               so they do not combine with `&=`; this plugin's own `eval` above ignores
               `active` for the same reason -- the caller masks the result. */
            return dr::select((si.wavelengths >= m_range.x()) &&
                                  (si.wavelengths <= m_range.y()),
                              Wavelength(1.f / (m_range.y() - m_range.x())),
                              Wavelength(0.f));
        } else {
            DRJIT_MARK_USED(si);
            NotImplementedError("pdf");
        }
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f & /*si*/,
                    const Wavelength & sample, Mask /*active*/) const override {
        if constexpr (is_spectral_v<Spectrum>) {
            return { m_range.x() +
                         (m_range.y() - m_range.x()) * sample,
                     m_value * (m_range.y() - m_range.x()) };
        } else {
            DRJIT_MARK_USED(sample);
            return { dr::empty<Wavelength>(), m_value };
        }
    }

    Float mean() const override { return m_value; }

    ScalarVector2f wavelength_range() const override {
        return m_range;
    }

    ScalarFloat spectral_resolution() const override {
        return 0.f;
    }

    ScalarFloat max() const override {
        return dr::slice(dr::max(m_value));
    }

    std::string to_string() const override {
        return tfm::format("UniformSpectrum[value=%f]", m_value);
    }

    MI_DECLARE_CLASS(UniformSpectrum)
private:
    Float m_value;
    ScalarVector2f m_range;

    MI_TRAVERSE_CB(SurfaceField, m_value)
};

MI_EXPORT_PLUGIN(UniformSpectrum)
NAMESPACE_END(mitsuba)
