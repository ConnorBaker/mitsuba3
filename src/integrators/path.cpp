#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path:

Path tracer (:monosp:`path`)
----------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
     corresponds to :math:`\infty`). A value of 1 will only render directly
     visible light sources. 2 will lead to single-bounce (direct-only)
     illumination, and so on. (Default: -1)

 * - transparent_max_depth
   - |int|
   - A separate budget for :monosp:`null` (transparent) interactions, matching Blender's
     :monosp:`transparent_max_bounces`. Passing through a transparent surface is not a
     scattering event, so with this set it is counted here instead of against
     :monosp:`max_depth`. (Default: -1, i.e. charge null interactions to :monosp:`max_depth`
     as before)

 * - diffuse_max_depth, glossy_max_depth, transmission_max_depth
   - |int|
   - Per-lobe bounce budgets, matching Blender's :monosp:`diffuse_bounces`,
     :monosp:`glossy_bounces` and :monosp:`transmission_bounces`. They partition
     reflect-vs-transmit first, exactly as Cycles does: a transmission is charged to
     :monosp:`transmission_max_depth` and to nothing else, so a *diffuse* transmission
     (translucency) does not spend the diffuse budget. Exceeding any budget does not stop
     the path immediately -- the next surface is still shaded, its emission added and its
     direct lighting sampled, and the path ends there. (Default: -1 each, i.e. unlimited)

 * - rr_depth
   - |int|
   - Specifies the path depth, at which the implementation will begin to use
     the *russian roulette* path termination criterion. For example, if set to
     1, then path generation may randomly cease after encountering directly
     visible surfaces. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer and is a **good default choice**
when there is no strong reason to prefer another method.

To use the path tracer appropriately, it is instructive to know roughly how
it works: its main operation is to trace many light paths using *random walks*
starting from the sensor. A single random walk is shown below, which entails
casting a ray associated with a pixel in the output image and searching for
the first visible intersection. A new direction is then chosen at the intersection,
and the ray-casting step repeats over and over again (until one of several
stopping criteria applies).

.. image:: ../../resources/data/docs/images/integrator/integrator_path_figure.png
    :width: 95%
    :align: center

At every intersection, the path tracer tries to create a connection to
the light source in an attempt to find a *complete* path along which
light can flow from the emitter to the sensor. This of course only works
when there is no occluding object between the intersection and the emitter.

This directly translates into a category of scenes where a path tracer can be
expected to produce reasonable results: this is the case when the emitters are
easily "accessible" by the contents of the scene. For instance, an interior
scene that is lit by an area light will be considerably harder to render when
this area light is inside a glass enclosure (which effectively counts as an
occluder).

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally
relies on multiple importance sampling to combine BSDF and emitter samples. The
main difference in comparison to the former plugin is that it considers light
paths of arbitrary length to compute both direct and indirect illumination.

.. note:: This integrator does not handle participating media

.. tabs::
    .. code-tab::  xml
        :name: path-integrator

        <integrator type="path">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'path',
        'max_depth': 8

 */

template <typename Float, typename Spectrum>
class PathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    PathIntegrator(const Properties &props) : Base(props) {
        /* A separate budget for NULL interactions, which is how Blender/Cycles counts.
           `max_depth` counts scattering EVENTS there; passing through a transparent surface
           is not one, and gets its own `transparent_max_bounces`. Mitsuba has historically
           charged both to `max_depth`, which leaves a converter two bad options: spend the
           light's bounce budget on transparent geometry (too dark), or add the two budgets
           together (too bright -- measured at +20% on a high-albedo interior, because the
           extra allowance is spent on REAL bounces wherever the path meets no transparent
           surface). Neither is what Blender does; this is.

           The default is -1, meaning "no separate budget": null interactions are charged to
           `max_depth` exactly as before, so no existing scene changes. */
        m_transparent_max_depth = props.get<int>("transparent_max_depth", -1);
        m_separate_null_budget = m_transparent_max_depth >= 0;

        /* The same idea, for the rest of Blender's budget set. Cycles does not have one
           bounce limit, it has five, and a converter that exports only `max_bounces` drops
           four of them silently -- which at Blender's own defaults is not a corner case:
           `max_bounces` is 12 while `diffuse_bounces` and `glossy_bounces` are 4, so on a
           typical interior the limit that actually binds is one of the ones being dropped.

           The counting rule is Cycles' own, from `path_state_next` in
           `intern/cycles/kernel/integrator/path_state.h`, and it is worth stating exactly
           because the obvious guess is wrong: the counters partition REFLECT-vs-TRANSMIT
           first, and only then split reflection into diffuse and glossy. A transmission
           increments `transmission_bounce` and NOTHING else -- so a DIFFUSE transmission
           (translucency) is charged to the transmission budget, never the diffuse one.

           Each is -1 by default, meaning unlimited, so a scene that names none of them is
           rendered exactly as before. */
        m_diffuse_max_depth      = props.get<int>("diffuse_max_depth", -1);
        m_glossy_max_depth       = props.get<int>("glossy_max_depth", -1);
        m_transmission_max_depth = props.get<int>("transmission_max_depth", -1);

        m_lobe_budgets = m_transparent_max_depth >= 0 || m_diffuse_max_depth >= 0 ||
                         m_glossy_max_depth >= 0 || m_transmission_max_depth >= 0;
    }

    std::pair<Spectrum, Bool> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        PreliminaryIntersection3f pi  = dr::zeros<PreliminaryIntersection3f>();
        UInt32 depth                  = 0;
        // Per-lobe budgets (\ref m_lobe_budgets); all inert unless one was configured.
        UInt32 null_depth             = 0;
        UInt32 diffuse_depth          = 0;
        UInt32 glossy_depth           = 0;
        UInt32 transmission_depth     = 0;
        /* Raised when a budget runs out, spent one iteration LATER -- see the stopping
           criterion, where the reason it cannot be spent immediately is set out. */
        Bool   stop_next              = false;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray = !m_hide_emitters && (scene->environment() != nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.
        */
        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            UInt32 null_depth;
            UInt32 diffuse_depth;
            UInt32 glossy_depth;
            UInt32 transmission_depth;
            Bool stop_next;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Bool active;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta, depth, \
                null_depth, diffuse_depth, glossy_depth, transmission_depth, stop_next,
                valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                active, sampler)
        } ls = {
            ray,
            pi,
            throughput,
            result,
            eta,
            depth,
            null_depth,
            diffuse_depth,
            glossy_depth,
            transmission_depth,
            stop_next,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            active,
            sampler
        };

        // First bounce is usually coherent - don't reorder threads
        if (unlikely(scene->has_ray_visibility_masks())) {
            // Blender-style per-object ray visibility (\ref RayVisibility): skip the shapes
            // the CAMERA cannot see. The ray comes back advanced past them, because `pi.t`
            // is measured along whichever ray the backend was finally handed.
            std::tie(ls.pi, ls.ray) = scene->ray_intersect_preliminary_visible(
                ls.ray, UInt32((uint32_t) RayVisibility::Camera),
                /* coherent = */ true, ls.active);
        } else {
            ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                     /* coherent = */ true,
                                                     /* reorder = */ false,
                                                     /* reorder_hint = */ 0,
                                                     /* reorder_hint_bits = */ 0,
                                                     ls.active);
        }

        // ---------------------- Hide area emitters ----------------------

        /* dr::any_or() checks for active entries in the provided boolean
           array. JIT/Megakernel modes can't do this test efficiently as
           each Monte Carlo sample runs independently. In this case,
           dr::any_or<..>() returns the template argument (true) which means
           that the 'if' statement is always conservatively taken. */

        if (m_hide_emitters && dr::any_or<true>(ls.depth == 0u)) {
            // Did we hit an area emitter? If so, skip all area emitters along this ray
            Mask skip_emitters = ls.pi.is_valid() &&
                                 (ls.pi.shape->emitter() != nullptr) &&
                                 ls.active;

            if (dr::any_or<true>(skip_emitters)) {
                SurfaceInteraction3f si = ls.pi.compute_surface_interaction(
                    ls.ray, +RayFlags::Minimal, skip_emitters);
                Ray3f ray = si.spawn_ray(ls.ray.d);
                PreliminaryIntersection3f pi_after_skip =
                    Base::skip_area_emitters(scene, ray, true, skip_emitters);
                dr::masked(ls.pi, skip_emitters) = pi_after_skip;
            }
        }

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, bsdf_ctx](LoopState& ls) {

            /* dr::while_loop implicitly masks all code in the loop using the
               'active' flag, so there is no need to pass it to every function */

            // Fill out all information of the interaction
            SurfaceInteraction3f si =
                ls.pi.compute_surface_interaction(ls.ray, +RayFlags::All);

            // ---------------------- Direct emission ----------------------

            if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                DirectionSample3f ds(scene, si, ls.prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!ls.prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds,
                                                          !ls.prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                ls.result = spec_fma(
                    ls.throughput,
                    ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.f) * mis_bsdf,
                    ls.result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next)) {
                ls.active = active_next;
                ls.valid_ray |= (si.emitter(scene) != nullptr) && !m_hide_emitters;
                return; // early exit for scalar mode
            }

            BSDFPtr bsdf = si.bsdf(ls.ray);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, ls.sampler->next_2d(), true, active_em);
                active_em &= (ds.pdf != 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(ds.pdf != 0, em_val / ds.pdf, 0);
                }

                wo = si.to_local(ds.d);
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1 = ls.sampler->next_1d();
            Point2f sample_2 = ls.sampler->next_2d();

            auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight]
                = bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1, sample_2);

            // --------------- Emitter sampling contribution ----------------

            if (dr::any_or<true>(active_em)) {
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                ls.result[active_em] = spec_fma(
                    ls.throughput, bsdf_val * em_weight * mis_em, ls.result);
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ls.ray)) {
                ls.ray = dr::detach<true>(ls.ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ls.ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, ls.active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            ls.throughput *= bsdf_weight;
            ls.eta *= bsdf_sample.eta;
            ls.valid_ray |= ls.active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            ls.prev_si = Interaction3f(si);
            ls.prev_bsdf_pdf = bsdf_sample.pdf;
            ls.prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            /* Classify the lobe that was just sampled. ONE classification, used by both the
               bounce budgets and the ray-visibility continuation below -- but note that they
               read it differently, and Cycles' `path_state_next` is explicit about the
               difference:

                 * the BOUNCE COUNTERS partition reflect-vs-transmit first, so the three are
                   mutually exclusive and a diffuse TRANSMISSION is charged to transmission;
                 * the RAY VISIBILITY bits are a UNION -- a transmission sets TRANSMIT *in
                   addition to* DIFFUSE or GLOSSY, so a diffuse transmission ray carries two
                   bits and is stopped by a shape that hides either one.

               A `null` interaction is neither: it is a pass-through, not a scattering event,
               which is the whole reason it gets its own budget. */
            Mask lobe_null    = has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);
            Mask lobe_transmit = has_flag(bsdf_sample.sampled_type, BSDFFlags::Transmission);
            Mask lobe_diffuse = has_flag(bsdf_sample.sampled_type, BSDFFlags::Diffuse);
            Mask at_surface   = si.is_valid();

            /* Charge the interaction to the right budget. Without a separate null budget the
               historical behaviour is reproduced exactly: everything lands on `depth`. */
            if (m_separate_null_budget) {
                dr::masked(ls.null_depth, at_surface && lobe_null) += 1;
                dr::masked(ls.depth, at_surface && !lobe_null) += 1;
            } else {
                dr::masked(ls.depth, at_surface) += 1;
            }

            if (m_lobe_budgets) {
                Mask scattered = at_surface && !lobe_null;
                dr::masked(ls.transmission_depth, scattered && lobe_transmit) += 1;
                dr::masked(ls.diffuse_depth,
                           scattered && !lobe_transmit && lobe_diffuse) += 1;
                dr::masked(ls.glossy_depth,
                           scattered && !lobe_transmit && !lobe_diffuse) += 1;
            }

            Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
            Mask rr_active = ls.depth >= m_rr_depth,
                 rr_continue = ls.sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            ls.active = active_next && (!rr_active || rr_continue) &&
                        (throughput_max != 0.f);

            /* Budget termination, in two steps, because a budget running out does NOT stop
               the path where it ran out. Cycles raises `PATH_RAY_TERMINATE_ON_NEXT_SURFACE`,
               and `integrate_surface_terminate` is consulted only AFTER the next surface has
               been shaded -- its emission added, its direct lighting sampled. So the flag
               raised at the bottom of one iteration is spent at the bottom of the NEXT one,
               and that ordering is the whole content of `stop_next`. Testing the counters
               directly here instead would drop a surface's emission and its NEE contribution,
               which is a darkening the size of one whole bounce.

               `>=` and not `>`: Cycles compares the count AFTER incrementing it. */
            if (m_lobe_budgets) {
                ls.active &= !ls.stop_next;

                Mask exhausted = false;
                if (m_transparent_max_depth >= 0)
                    exhausted |= ls.null_depth >= (uint32_t) m_transparent_max_depth;
                if (m_diffuse_max_depth >= 0)
                    exhausted |= ls.diffuse_depth >= (uint32_t) m_diffuse_max_depth;
                if (m_glossy_max_depth >= 0)
                    exhausted |= ls.glossy_depth >= (uint32_t) m_glossy_max_depth;
                if (m_transmission_max_depth >= 0)
                    exhausted |= ls.transmission_depth >= (uint32_t) m_transmission_max_depth;

                ls.stop_next |= exhausted;
            }

            if (unlikely(scene->has_ray_visibility_masks())) {
                /* Blender-style per-object ray visibility (\ref RayVisibility). Which switch
                   the continuation answers to is decided by the lobe that was just sampled,
                   which is Blender's own division: everything that is not diffuse --
                   including a Delta specular lobe -- counts as glossy, and a transmission
                   adds its bit ON TOP rather than replacing it. Cycles' `path_state_next`
                   ORs `PATH_RAY_VISIBILITY_TRANSMIT` into a visibility that already carries
                   DIFFUSE or GLOSSY, so a diffuse transmission ray answers to BOTH switches
                   and is stopped by a shape that hides either. The type varies per lane,
                   which is why it is a `UInt32` rather than a constant. */
                UInt32 next_type =
                    dr::select(lobe_diffuse, UInt32((uint32_t) RayVisibility::Diffuse),
                               UInt32((uint32_t) RayVisibility::Glossy)) |
                    dr::select(lobe_transmit,
                               UInt32((uint32_t) RayVisibility::Transmission), UInt32(0u));

                std::tie(ls.pi, ls.ray) = scene->ray_intersect_preliminary_visible(
                    ls.ray, next_type, /* coherent = */ false, ls.active);
            } else {
            // Reorder threads based on the shape they hit
            ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                     /* coherent = */ false,
                                                     /* reorder = */ jit_flag(JitFlag::LoopRecord),
                                                     /* reorder_hint = */ 0,
                                                     /* reorder_hint_bits = */ 0,
                                                     ls.active);
            }
        });

        return {
            /* spec  = */ dr::select(ls.valid_ray, ls.result, 0.f),
            /* valid = */ ls.valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathIntegrator[\n"
            "  max_depth = %u,\n"
            "  transparent_max_depth = %i,\n"
            "  diffuse_max_depth = %i,\n"
            "  glossy_max_depth = %i,\n"
            "  transmission_max_depth = %i,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_transparent_max_depth, m_diffuse_max_depth,
            m_glossy_max_depth, m_transmission_max_depth, m_rr_depth);
    }

    /// Separate bounce budget for `null` (transparent) interactions; -1 disables it
    int m_transparent_max_depth = -1;
    bool m_separate_null_budget = false;

    /// Blender's remaining per-lobe budgets; -1 each disables that one
    int m_diffuse_max_depth = -1;
    int m_glossy_max_depth = -1;
    int m_transmission_max_depth = -1;
    /// True when ANY of the four per-lobe budgets is in use
    bool m_lobe_budgets = false;

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS(PathIntegrator)
};

MI_EXPORT_PLUGIN(PathIntegrator)
NAMESPACE_END(mitsuba)
