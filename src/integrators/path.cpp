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
     :monosp:`max_depth`. Cycles stores :monosp:`transparent_max_bounce` verbatim (no
     plus-one, unlike its lobe budgets), so pass Blender's value unchanged. (Default: -1,
     i.e. charge null interactions to :monosp:`max_depth` as before)

     Cycles' remaining per-lobe budgets (:monosp:`diffuse_bounces`,
     :monosp:`glossy_bounces`, :monosp:`transmission_bounces`) are deliberately NOT
     implemented: truncating transport per lobe is a nonphysical variance control. The
     constructor refuses those property names rather than ignoring them; for Cycles
     parity, raise the Blender scene's per-lobe bounces to :monosp:`max_bounces`.

 * - rr_depth
   - |int|
   - Specifies the path depth, at which the implementation will begin to use
     the *russian roulette* path termination criterion. For example, if set to
     1, then path generation may randomly cease after encountering directly
     visible surfaces. (Default: 5)

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
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
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

        /* Cycles' per-sample radiance clamp; see \ref clamp_light. Blender's UI values,
           0 meaning "off", with Cycles' own factor of three folded in here so that the
           number written in the .blend is the number written here. Both default to 0 so
           that a scene which does not ask for the clamp gets an unbiased estimator --
           Blender's own default for the indirect side is 10.0, not 0, and it is the
           EXPORTER's job to carry that across rather than this plugin's job to assume it. */
        ScalarFloat clamp_direct   = props.get<ScalarFloat>("clamp_direct", 0.f);
        ScalarFloat clamp_indirect = props.get<ScalarFloat>("clamp_indirect", 0.f);
        if (clamp_direct < 0.f || clamp_indirect < 0.f)
            Throw("PathIntegrator: clamp_direct and clamp_indirect must be non-negative "
                  "(0 disables the clamp); got %f and %f", clamp_direct, clamp_indirect);
        m_clamp_direct   = clamp_direct   == 0.f ? dr::Infinity<ScalarFloat>
                                                 : clamp_direct * 3.f;
        m_clamp_indirect = clamp_indirect == 0.f ? dr::Infinity<ScalarFloat>
                                                 : clamp_indirect * 3.f;
        m_clamp_enabled  = clamp_direct != 0.f || clamp_indirect != 0.f;

        /* Cycles' "filter glossy" (Blender UI: Sampling > Advanced > Filter Glossy,
           `cycles.blur_glossy`, factory default 1.0). Taken here in the UI's units and
           inverted the way Cycles does in `scene/integrator.cpp`:
             kintegrator->filter_glossy = (filter_glossy == 0.0f) ? FLT_MAX
                                                                  : 1.0f / filter_glossy;
           Zero disables it, which is also this plugin's default -- so a scene that does
           not ask for it renders exactly as it did before. */
        ScalarFloat blur_glossy = props.get<ScalarFloat>("blur_glossy", 0.f);
        if (blur_glossy < 0.f)
            Throw("PathIntegrator: blur_glossy must be non-negative (0 disables filter "
                  "glossy); got %f", blur_glossy);
        m_filter_glossy_enabled = blur_glossy != 0.f;
        m_filter_glossy = m_filter_glossy_enabled ? 1.f / blur_glossy
                                                  : dr::Infinity<ScalarFloat>;

        /* Cycles' per-lobe bounce budgets are deliberately not implemented (see the
           plugin docstring). Refuse the keys by name: silently ignoring them would
           render a different image than the scene asked for. */
        for (const char *removed : { "diffuse_max_depth", "glossy_max_depth",
                                     "transmission_max_depth" })
            if (props.has_property(removed))
                Throw("PathIntegrator: \"%s\" was removed (a nonphysical Cycles parity "
                      "shim). For Cycles parity, raise the Blender scene's per-lobe "
                      "bounces to max_bounces instead.", removed);
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
        // Null (transparent) crossings, budgeted separately (\ref m_separate_null_budget).
        UInt32 null_depth             = 0;
        /* Raised when a budget runs out, spent one iteration LATER -- see the stopping
           criterion, where the reason it cannot be spent immediately is set out. */
        Bool   stop_next              = false;

        Mask valid_ray = false;

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        /* Smallest BSDF sampling density along the path so far -- Cycles'
           `INTEGRATOR_STATE(state, path, min_ray_pdf)`, initialised to FLT_MAX in
           `kernel/integrator/path_state.h`. Only read when filter glossy is enabled. */
        Float         min_ray_pdf     = dr::Infinity<Float>;
        BSDFContext   bsdf_ctx;

        // Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
        // mode, and it generates either a megakernel (default) or
        // wavefront-style renderer in JIT variants. This can be controlled by
        // passing the '-W' command line flag to the mitsuba binary or
        // enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.
        struct LoopState {
            Ray3f ray;
            PreliminaryIntersection3f pi;
            Spectrum throughput;
            Spectrum result;
            Float eta;
            UInt32 depth;
            UInt32 null_depth;
            Bool stop_next;
            Mask valid_ray;
            Interaction3f prev_si;
            Float prev_bsdf_pdf;
            Bool prev_bsdf_delta;
            Float min_ray_pdf;
            /* Which RayMask switch this ray currently answers to. It has to be CARRIED
               rather than recomputed each bounce, because a NULL interaction must not change
               it -- see the note where it is updated. */
            UInt32 ray_visibility;
            Bool active;
            Sampler* sampler;

            DRJIT_STRUCT(LoopState, ray, pi, throughput, result, eta, depth, \
                null_depth, stop_next,
                valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta, min_ray_pdf,
                ray_visibility, active, sampler)
        } ls = {
            ray,
            pi,
            throughput,
            result,
            eta,
            depth,
            null_depth,
            stop_next,
            valid_ray,
            prev_si,
            prev_bsdf_pdf,
            prev_bsdf_delta,
            min_ray_pdf,
            UInt32((uint32_t) RayMask::Camera),
            active,
            sampler
        };

        // First bounce is usually coherent - don't reorder threads. The
        // camera mask (ls.ray_visibility's initial value) hides emitters
        // marked as invisible and shapes with `visible_camera = false`.
        ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                 /* coherent = */ true,
                                                 /* reorder = */ false,
                                                 /* reorder_hint = */ 0,
                                                 /* reorder_hint_bits = */ 0,
                                                 ls.active,
                                                 ls.ray_visibility);

        dr::tie(ls) = dr::while_loop(dr::make_tuple(ls),
            [](const LoopState& ls) { return ls.active; },
            [this, scene, bsdf_ctx](LoopState& ls) {

            // dr::while_loop implicitly masks all code in the loop using the
            // 'active' flag, so there is no need to pass it to every function

            // Fill out all information of the interaction
            SurfaceInteraction3f si = scene->compute_surface_interaction(
                ls.ray, ls.pi, +RayFlags::Default);

            /* Filter glossy: hand this shade point the roughness floor implied by how
               improbable the path that reached it was. Cycles does the same thing at the
               same moment, in `surface_shader_prepare_closures`
               (`kernel/integrator/surface_shader.h`), except that it MUTATES the closures
               it has just allocated -- which a Mitsuba BSDF plugin, being a shared
               immutable object, cannot do. Carrying the floor on the interaction instead
               puts it exactly where every `eval` / `sample` / `pdf` already looks.

               `min_ray_pdf` starts at infinity, so `blur_pdf` is infinite on the first
               vertex and the branch below leaves `min_alpha` at zero -- a camera-visible
               surface is never blurred, matching Cycles. */
            if (m_filter_glossy_enabled) {
                Float blur_pdf = m_filter_glossy * ls.min_ray_pdf;
                si.min_alpha = dr::select(blur_pdf < 1.f,
                                          dr::sqrt(1.f - blur_pdf) * 0.5f, Float(0.f));
            }

            // ---------------------- Direct emission ----------------------

            /* Ray mask of the trace that produced si, to handle hidden emitters.
               `ls.ray_visibility` is exactly that mask: it was Camera for the first
               trace and is reassigned at the bottom of the loop before every other,
               so no depth-based reconstruction is needed. */
            UInt32 ray_mask = ls.ray_visibility;

            EmitterPtr emitter = si.emitter(scene, true, ray_mask);
            ls.valid_ray |= emitter != nullptr;

            if (dr::any_or<true>(emitter != nullptr)) {
                DirectionSample3f ds(scene, si, ls.prev_si, ray_mask);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!ls.prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(ls.prev_si, ds,
                                                          !ls.prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);

                /* Accumulate, being careful with polarization (see spec_fma).

                   The clamp's bounce index for emission found by a BSDF sample is
                   Cycles' `path.bounce - 1` (`film_write_surface_emission`). `ls.depth`
                   is that `bounce` -- the number of scattering events already performed
                   -- so "indirect" is `depth - 1 > 0`, i.e. `depth >= 2`. Written as a
                   comparison rather than a subtraction because `depth` is unsigned and
                   the direct case is exactly the one that would underflow. */
                Spectrum emitted = clamp_light(
                    ls.throughput * emitter->eval(si, ls.prev_bsdf_pdf > 0.f) * mis_bsdf,
                    ls.depth >= 2);
                ls.result += emitted;
            }

            // Continue tracing the path at this point?
            /* `stop_next` carries an exhausted transparent budget forward by one vertex, and
               it belongs HERE, folded into `active_next`, rather than after the bounce.
               Cycles' terminate flags are all covered by `PATH_RAY_TERMINATE`, and
               `surface_shader_prepare_closures` reads that as `max_closures = 0`: the next
               surface is reached with NO closures at all. Its emission is still added, but
               there is nothing to sample a light against and nothing to scatter with -- no
               direct lighting, no bounce. That is exactly what `active_next` already means
               for `max_depth`, so the two truncations are the same truncation and are
               spelled the same way.

               Placing it after the bounce instead -- the obvious reading of "terminate on
               the NEXT surface" -- grants the terminal vertex a direct-lighting contribution
               Cycles never computes, which on an interior is a visible over-brightening. */
            Bool active_next = (ls.depth + 1 < m_max_depth) && si.is_valid() && !ls.stop_next;

            if (dr::none_or<false>(active_next)) {
                ls.active = active_next;
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
                /* The visibility test is handed to `null_shadow_transmittance` below
                   whenever the scene contains a pass-through boundary, because `ray_test`
                   -- which is what `true` asks for here -- cannot see through one. */
                bool march_shadow = scene->has_null_bsdfs();
                std::tie(ds, em_weight) = scene->sample_emitter_direction(
                    si, ls.sampler->next_2d(), !march_shadow, active_em);
                active_em &= (ds.pdf != 0.f);

                // Given the detached emitter sample, recompute its contribution
                // with AD to enable light source optimization.
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(ds.pdf != 0, em_val / ds.pdf, 0);
                }

                if (unlikely(march_shadow)) {
                    Spectrum tr =
                        null_shadow_transmittance(scene, si, ds.p, active_em);
                    em_weight *= tr;
                    active_em &= dr::any(unpolarized_spectrum(tr) != 0.f);
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

                /* Accumulate, being careful with polarization (see spec_fma).

                   For a light sampled directly, Cycles clamps on `shadow_path.bounce`
                   (`film_write_direct_light`), which is `path.bounce` at the moment the
                   shadow ray is cast -- `ls.depth` here. So "indirect" is `depth >= 1`,
                   one lower than the emission site above. The two agree physically: NEE
                   at the first hit and BSDF-sampled emission after one bounce describe
                   the same single-scattering path, and both are classified as direct. */
                Spectrum sampled = clamp_light(
                    ls.throughput * bsdf_val * em_weight * mis_em, ls.depth >= 1);
                ls.result[active_em] = ls.result + sampled;
            }

            // ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            // When the path tracer is differentiated, we must be careful that
            // the generated Monte Carlo samples are detached (i.e. don't track
            // derivatives) to avoid bias resulting from the combination of moving
            // samples and discontinuous visibility. We need to re-evaluate the
            // BSDF differentiably with the detached sample in that case.
            if (dr::grad_enabled(ls.ray)) {
                ls.ray = dr::detach(ls.ray);

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

            /* Information about the current vertex needed by the next iteration -- but ONLY
               when this vertex actually scattered. A `null` crossing is a pass-through, and
               overwriting the MIS bookkeeping with it corrupts the next emitter hit in two
               separate ways:

                 * `prev_si` becomes the PANE rather than the last real scattering vertex, so
                   `pdf_emitter_direction` converts the emitter's area density to solid angle
                   using the wrong distance. A pane hovering just in front of a lamp reports
                   an almost-zero pdf where the true shading point reports a large one.
                 * `prev_bsdf_pdf` becomes the null lobe's own pdf (1), which is not a
                   scattering density at all and belongs to no sampling strategy.

               `mis_weight(prev_bsdf_pdf, em_pdf)` then returns ~1 instead of that strategy's
               proper share, so a BSDF-sampled emitter hit reached through a clear pane is
               counted at FULL weight while NEE also counts it -- the emitter is paid for
               twice. MEASURED as exactly that: a furnace box read 2.0x Cycles on every
               pane case (1.9943 with one pane, 2.0052/2.0055/2.0059 with six/seven/eight)
               and exactly 1.0000 with the pane removed. The factor is constant in the pane
               COUNT, which is what identifies it as a once-per-path MIS error rather than a
               per-crossing transmittance one.

               This also explains why the bug hid for so long. Before shadow rays were
               marched through null boundaries, NEE was dead behind a pane (`ray_test` cannot
               see through one) and this over-weighted BSDF hit was silently standing in for
               it -- two errors nearly cancelling, for a furnace ratio of 0.9725 that looked
               like a 2.7% rounding issue rather than two independent defects.

               Cycles guards the same bookkeeping the same way, and `min_ray_pdf` below
               already carried this exact guard: `if (!(label & LABEL_TRANSPARENT))`. */
            Mask real_scatter = !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);
            dr::masked(ls.prev_si, real_scatter) = Interaction3f(si);
            dr::masked(ls.prev_bsdf_pdf, real_scatter) = bsdf_sample.pdf;
            dr::masked(ls.prev_bsdf_delta, real_scatter) =
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            /* Cycles: `if (!(label & LABEL_TRANSPARENT)) min_ray_pdf = fminf(
                 unguided_bsdf_pdf, min_ray_pdf);` (`kernel/integrator/shade_surface.h`).
               A pass-through is not a scattering event and must not tighten the bound --
               otherwise a stack of transparent surfaces would blur everything behind it. */
            if (m_filter_glossy_enabled) {
                /* `si.is_valid()` is DEFENSIVE, and unlike the Null test it is not
                   currently load-bearing -- said plainly because the first version of this
                   comment claimed otherwise. The reasoning that motivated it is sound as
                   far as it goes: `ls.active` here still holds the PREVIOUS iteration's
                   value (in vectorized mode it is only reassigned at the `ls.active =
                   active_next && ...` line below, whereas `active_next` is what carries
                   `si.is_valid()`), so an escaped lane does reach this line, and its
                   `bsdf_sample.pdf` is 0 -- which would pin the bound to 0, the MAXIMUM
                   blur. What the reasoning missed is that the same lane is killed at that
                   `ls.active` line before it can ever READ `min_ray_pdf` again, so the
                   corrupted bound is never used.

                   Measured rather than argued: building with and without this clause gives
                   byte-identical images (6/6 sha256 matches over open scenes chosen to
                   escape as much as possible, half with a constant envmap so escaping rays
                   carry energy). Kept anyway -- it costs one mask operation and it is the
                   difference between "correct" and "correct as long as nobody reorders the
                   termination logic". Cycles cannot reach the case at all, because it only
                   arrives at the update after a real surface hit. */
                Mask scattered = ls.active && si.is_valid() &&
                    !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);
                dr::masked(ls.min_ray_pdf, scattered) =
                    dr::minimum(bsdf_sample.pdf, ls.min_ray_pdf);
            }

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
            Mask lobe_null     = has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);
            Mask lobe_transmit = has_flag(bsdf_sample.sampled_type, BSDFFlags::Transmission);
            Mask lobe_diffuse  = has_flag(bsdf_sample.sampled_type, BSDFFlags::Diffuse);
            Mask at_surface    = si.is_valid();

            /* Charge the interaction to the right budget. Without a separate null budget the
               historical behaviour is reproduced exactly: everything lands on `depth`. */
            if (m_separate_null_budget) {
                dr::masked(ls.null_depth, at_surface && lobe_null) += 1;
                dr::masked(ls.depth, at_surface && !lobe_null) += 1;
            } else {
                dr::masked(ls.depth, at_surface) += 1;
            }

            Mask charged_null = at_surface && lobe_null;

            Float throughput_max = dr::max(unpolarized_spectrum(ls.throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::square(ls.eta), .95f);
            Mask rr_active = ls.depth >= m_rr_depth,
                 rr_continue = ls.sampler->next_1d() < rr_prob;

            // Differentiable variants of the renderer require the russian
            // roulette sampling weight to be detached to avoid bias. This is a
            // no-op in non-differentiable variants.
            ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            ls.active = active_next && (!rr_active || rr_continue) &&
                        (throughput_max != 0.f);

            /* Raise the flag; `active_next` above spends it on the next iteration. The
               budget running out does not stop the path where it ran out -- the next
               surface is still reached and its emission still collected, which is Cycles'
               `TERMINATE_ON_NEXT_SURFACE`, the flag its transparent budget raises.

               `>=` and not `>`: Cycles compares the count AFTER incrementing it
               (`kernel/integrator/path_state.h`). It stores `transparent_max_bounce`
               verbatim (no plus-one, unlike the lobe budgets it also carries -- which is
               why 0 and 1 behave alike there; Cycles' own FIXME in `path_state.h` calls
               that out). The comparison is gated on having just CHARGED the counter,
               which is where Cycles puts it: the budget can only ever be tripped by a
               crossing of its own kind. */
            if (m_separate_null_budget)
                ls.stop_next |= charged_null &&
                                ls.null_depth >= (uint32_t) m_transparent_max_depth;

            /* Blender-style per-object ray visibility (\ref RayMask). Which switch the
               continuation answers to is decided by the lobe that was just sampled, which
               is Blender's own division: everything that is not diffuse -- including a
               Delta specular lobe -- counts as glossy, and a transmission adds its bit ON
               TOP rather than replacing it. Cycles' `path_state_next` ORs
               `PATH_RAY_VISIBILITY_TRANSMIT` into a visibility that already carries
               DIFFUSE or GLOSSY, so a diffuse transmission ray answers to BOTH switches
               and is stopped by a shape that hides either. The type varies per lane, which
               is why it is a `UInt32` rather than a constant. */
            UInt32 scattered_type =
                dr::select(lobe_diffuse, UInt32((uint32_t) RayMask::Diffuse),
                           UInt32((uint32_t) RayMask::Glossy)) |
                dr::select(lobe_transmit,
                           UInt32((uint32_t) RayMask::Transmission), UInt32(0u));

            /* A NULL interaction does NOT reclassify the ray. Cycles is explicit about
               this in `path_state_next` (intern/cycles/kernel/integrator/path_state.h):

                   ray through transparent keeps same flags from previous ray and is
                   not counted as a regular bounce, transparent has separate max

               -- it ORs in `PATH_RAY_TRANSPARENT` and returns early, never clearing
               `PATH_RAY_CAMERA` and never setting DIFFUSE / GLOSSY / TRANSMIT. So a
               camera ray that crosses a transparent surface is STILL a camera ray.

               Recomputing the type from the sampled lobe made a null crossing turn a
               camera ray into a transmission one, which un-hid every shape the camera is
               not supposed to see as soon as anything transparent stood in front of it.
               Measured in a closed grey box with one lamp behind a single Transparent
               pane: Mitsuba's p99 and max were 15.99766 and 16.07686 -- exactly the
               emitter's radiance -- against a Cycles image whose maximum anywhere was
               0.30489, a 4.04x error in the mean. The same scene with the pane removed
               was already correct at 1.0003, so the pane was the whole difference.

               This is not a corner case for a Blender exporter: every area lamp is
               exported as geometry wearing a `null` BSDF, and window glass is routinely
               modelled the same way. */
            ls.ray_visibility = dr::select(lobe_null, ls.ray_visibility, scattered_type);

            // Reorder threads based on the shape they hit
            ls.pi = scene->ray_intersect_preliminary(ls.ray,
                                                     /* coherent = */ false,
                                                     /* reorder = */ jit_flag(JitFlag::LoopRecord),
                                                     /* reorder_hint = */ 0,
                                                     /* reorder_hint_bits = */ 0,
                                                     ls.active,
                                                     ls.ray_visibility);
        });

        return {
            /* spec  = */ dr::select(ls.valid_ray, ls.result, 0.f),
            /* valid = */ ls.valid_ray
        };
    }

    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathIntegrator[\n"
            "  max_depth = %u,\n"
            "  transparent_max_depth = %i,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_transparent_max_depth, m_rr_depth);
    }

    /// Separate bounce budget for `null` (transparent) interactions; -1 disables it
    int m_transparent_max_depth = -1;
    bool m_separate_null_budget = false;

    ScalarFloat m_clamp_direct   = dr::Infinity<ScalarFloat>;
    ScalarFloat m_clamp_indirect = dr::Infinity<ScalarFloat>;
    bool m_clamp_enabled = false;
    /// Cycles' `filter_glossy`, already inverted (1 / the UI's `blur_glossy`).
    ScalarFloat m_filter_glossy = dr::Infinity<ScalarFloat>;
    bool m_filter_glossy_enabled = false;

    /**
     * \brief March a shadow ray through `null` boundaries and return its transmittance.
     *
     * `Scene::ray_test` is a pure occlusion query: it cannot tell a pane of clear glass
     * from a wall, and reports Blender's Transparent BSDF as a hit like any other. Testing
     * NEE visibility with it therefore kills every direct-lighting contribution behind a
     * transparent surface. MEASURED on a furnace box with one Transparent pane between the
     * lamp and the room: `scene.ray_test()` across a lone `null` pane returns True, and the
     * rendered image sat 2.7% below Cycles with the tell of a dead estimator -- the tails
     * blown out in BOTH directions at 256 spp (mi min 0.01078 / max 0.44534 against cy
     * 0.03553 / 0.30489) while the no-pane control matched to 1.0001.
     *
     * The DEFICIT is the part worth explaining, because "NEE is blocked" on its own predicts
     * only extra variance: the BSDF-sampled hit on that same emitter is still weighted by
     * `mis_weight(bsdf_pdf, em_pdf)`, i.e. discounted for an NEE contribution that never
     * arrived. Half the estimator is missing and the surviving half is still paying its
     * share, so the result is biased LOW rather than merely noisy. That is why this cannot
     * be left to converge away.
     *
     * Cycles traces a real transparent shadow ray (`integrate_transparent_shadow`), which is
     * what is reproduced here: pass through anything with a Null lobe, multiplying by its
     * `eval_null_transmission` (1 for a plain `null`, the alpha for a `mask`), stop dead at
     * the first surface that has no Null lobe, and charge each crossing to
     * `transparent_max_depth` -- the SAME budget the main path spends, which is Cycles'
     * `transparent_max_bounces`. A ray that exceeds it is treated as blocked.
     *
     * `RayMask::Shadow` is threaded through so that per-object shadow visibility keeps
     * working (a shape with `visible_shadow = false` must not stop the march either);
     * this is the one place it belongs on this path, and the acceleration structures do
     * the filtering.
     *
     * Not run unless \ref Scene::has_null_bsdfs(), so a scene without a single pass-through
     * boundary keeps the one fast `ray_test` and pays nothing.
     */
    Spectrum null_shadow_transmittance(const Scene *scene,
                                       const SurfaceInteraction3f &ref,
                                       const Point3f &target, Mask active) const {
        struct ShadowState {
            Mask active;
            Ray3f ray;
            Spectrum transmittance;
            UInt32 crossings;

            DRJIT_STRUCT(ShadowState, active, ray, transmittance, crossings)
        } ss = { active, ref.spawn_ray_to(target), Spectrum(1.f), UInt32(0) };

        dr::tie(ss) = dr::while_loop(
            dr::make_tuple(ss),
            [](const ShadowState &ss) { return dr::detach(ss.active); },
            [this, scene, target](ShadowState &ss) {
                /* The Shadow mask: a shape hidden from shadow rays must not stop the
                   march either. Reduces to the unmasked query when no shape in the
                   scene carries a mask (All & Shadow != 0 for every ordinary shape). */
                PreliminaryIntersection3f pi = scene->ray_intersect_preliminary(
                    ss.ray, /* coherent = */ false, ss.active,
                    UInt32((uint32_t) RayMask::Shadow));
                SurfaceInteraction3f si = scene->compute_surface_interaction(
                    ss.ray, pi, +RayFlags::Default, ss.active);

                Mask hit = ss.active && si.is_valid();

                /* Early exit for scalar mode, and it is not an optimisation: with one lane
                   and no masking, `si.bsdf()` on an escaped ray dereferences a null shape
                   and segfaults. The JIT variants trace straight past this (`none_or<false>`
                   is false at trace time) and rely on the mask instead -- which is why the
                   GPU arm of the regression suite passed while every `scalar_rgb` case
                   crashed. Same shape as the guard the main loop already carries. */
                if (dr::none_or<false>(hit)) {
                    ss.active = false;
                    return;
                }

                BSDFPtr bsdf  = si.bsdf(ss.ray);
                Mask is_null  = hit && has_flag(bsdf->flags(), BSDFFlags::Null);

                /* Reaching a surface with no pass-through lobe is an ordinary occlusion:
                   the light is blocked outright, exactly as `ray_test` would have said. */
                dr::masked(ss.transmittance, hit && !is_null) = 0.f;

                Spectrum tr = bsdf->eval_null_transmission(si, is_null);
                tr = si.to_world_mueller(tr, si.wi, si.wi);
                dr::masked(ss.transmittance, is_null) *= tr;
                dr::masked(ss.crossings, is_null) += 1;

                /* `>` and not `>=`, unlike the main path's budget test: there the counter
                   is compared after incrementing and trips a `stop_next` that still lets
                   the NEXT surface be reached, so a budget of D buys D crossings. A shadow
                   ray has no such grace -- blocked is blocked -- so spelling it `>` is what
                   makes the two agree on the number of panes light gets through. */
                Mask budget_out = false;
                if (m_transparent_max_depth >= 0)
                    budget_out = is_null &&
                                 ss.crossings > (uint32_t) m_transparent_max_depth;
                dr::masked(ss.transmittance, budget_out) = 0.f;

                /* Re-aim at the emitter rather than re-using the direction: `spawn_ray_to`
                   recomputes `maxt`, so the march cannot overshoot the light. */
                dr::masked(ss.ray, is_null) = si.spawn_ray_to(target);

                ss.active = is_null && !budget_out &&
                            dr::any(unpolarized_spectrum(ss.transmittance) != 0.f);
            },
            "Path integrator transparent shadow ray");

        return ss.transmittance;
    }

    /**
     * \brief Cycles' per-sample radiance clamp (\c film_clamp_light).
     *
     * Blender ships this ON: \c sample_clamp_indirect defaults to 10.0, so an importer
     * that ignores it is not comparing against the estimator Cycles actually ran. It is
     * deliberately biased -- it exists to kill fireflies -- and the bias is not small
     * wherever a path can carry a very large contribution.
     *
     * Transcribed from \c intern/cycles/kernel/film/light_passes.h:
     *
     * \code
     *   const float limit = (bounce > 0) ? sample_clamp_indirect : sample_clamp_direct;
     *   const float sum = reduce_add(fabs(*L));
     *   if (sum > limit) { *L *= limit / sum; }
     * \endcode
     *
     * Three details are easy to get wrong and are all load-bearing. The test is on the
     * L1 SUM of the channels, not on any single channel and not on luminance. The whole
     * spectrum is then scaled by one common factor, so clamping shifts brightness but
     * never hue. And \c scene/integrator.cpp stores the UI value multiplied by three
     * (\c sample_clamp_direct * 3.0f), with zero mapped to \c FLT_MAX rather than to a
     * limit of zero -- which is why 0 means "off" instead of "block everything". The
     * factor of three is applied here so that this plugin's parameters carry the same
     * numbers as Blender's UI, and it is Cycles' RGB convention: in a spectral variant
     * the sum runs over wavelengths and the correspondence is no longer exact.
     *
     * The scale is computed from the unpolarized intensity and then applied to the full
     * Spectrum, so a Mueller matrix is attenuated rather than reinterpreted.
     */
    Spectrum clamp_light(const Spectrum &L, const Mask &indirect) const {
        if (!m_clamp_enabled)
            return L;

        UnpolarizedSpectrum us = unpolarized_spectrum(L);
        Float sum = 0.f;
        for (size_t i = 0; i < dr::size_v<UnpolarizedSpectrum>; ++i)
            sum += dr::abs(us[i]);

        Float limit = dr::select(indirect, Float(m_clamp_indirect), Float(m_clamp_direct));
        // `sum > limit` is false when limit is infinite, so a disabled side is a no-op
        // without a second branch; and sum == 0 cannot reach the division.
        Float scale = dr::select(sum > limit, limit / sum, 1.f);
        return L * scale;
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * Perform a Mueller matrix multiplication in polarized modes, and a
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
