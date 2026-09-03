# hsr/blender-parity -> hsr/master-port: feature-port log

The 84 fork commits (v3.9.0..4ded1df8) ported cluster by cluster onto the
master port stack (see STACK.md). Cluster map and dispositions follow the HSR
repo's `investigations/mitsuba_upstream_audit.md` (§2 delta table, §3
per-commit rows). One row per cluster; update the row when the cluster's
status changes, do not append duplicates.

Statuses: `pending` / `ported` (code landed) / `compiled` / `tested`.

| # | cluster | fork commits | status | notes |
|---|---|---|---|---|
| P1 | Plugin-registration infra + texture plugin tree | d67d0c51 (#1732) + b2b53c47, c604b572, 0b383600, a5f03f7b, 730126e5, eb4150b5, 16522460, be924a8b, 2d130628, 839c3f6c, deda5367, 77cef9d6, adad37b2, 492acb9f/2a557651/7dad744c (bump plugin) | tested (smoke) | See "P1 adaptations" below. 13/13 smoke checks on cuda_ad_rgb incl. variant-change reload; the final full-suite run (STACK.md) exercises the field registry with no failures beyond the pre-port baseline. |
| P2 | ad/bsdfs (blender_principled + lobes/tables) + BSDF C++ surface | e1bc2905 (#1822), b1fde2f3, 144aaac8, 95c07233, 638d4920, 8b94cda5, a8f6106a, 81fae57d, f1f01f3a, f6c8054f, 596da59a, 991a1844, fcca650e, 3ac0ff8f, 62936c42, b3c64fb7, faec9175 (rough* floors), 77f35439, 4b19696/8b94cda5 tests | tested | C++ (bsdf.h sampled_roughness_squared, interaction.h min_alpha, microfacet.h filter_glossy_alpha, six BSDF cpps, bindings) 3-way applied; 5 conflicts resolved against master's doc-conversion + instance_index/frame_flipped churn. Python tree verbatim + reload entry. principledhelpers.h delta dropped (upstream 366de608 already on master). Compiled at e4d60f01; all 10 suites (44 tests) pass on cuda_ad_rgb -- incl. test_bsdf_sample_struct on the new field, after the pre-compile run failed all 16 min_alpha readers exactly as predicted. |
| P3 | Emitters: area si.wi, spot, blender_lamp | 81ea8f2d (4ea28b4a superseded by upstream b9286db1), c3f60515, 4ed80deb, 4ded1df8, 011aa296 (retired into blender_lamp) | tested | area.cpp: only 81ea8f2d's three si.wi hunks re-applied by hand onto upstream's b9286db1 two-sided implementation (our 4ea28b4a arm dropped per the audit disposition); spot.cpp applied clean 3-way; blender_lamp + emitters registry verbatim + reload entry. Compiled at d856e1d7. All ported emitter tests pass on the P3 binary (cuda_ad_rgb + scalar_rgb where parametrized) after two upstream-churn test adaptations: test_area_two_sided uses upstream b9286db1's property/to_string spelling `twosided` (fork spelled it `two_sided`), and test_area_directional_radiance's WiCosine helper is ported to the Field API (mi.Field base + contract methods, eval -> eval_color3, register_field). Remaining failures in src/emitters/tests are all upstream test_sunsky missing resources/data/tests/sunsky reference EXRs -- unrelated. |
| P4 | Sensor/film/rfilter/bitmap | 217be112, f0a48397, f665f9e7 (blackman_harris), a89facc4 + 3d4a9faf (test) | tested | Compiled at ef5383de; test_specfilm (incl. test08 on both spectral variants) + test_blackman_harris + test_texture_bitmap + test_perspective all pass (73 passed, 4 skipped). thinlens/orthographic + 6 upstream cherry-picks already ON master: drop. sensor.cpp SRF inverse-PDF weight (217be112) applied clean; specfilm.cpp prepare_sample conflict resolved (master renamed the aggregate to `spec`; keep upstream's structure, alias `values`); blackman_harris rfilter + test copied verbatim + CMakeLists entry; fields/bitmap.cpp eval_3 mono->Color3 broadcast re-applied (channels==1 arm; the fork's eval_1_grad fix + test12 already on master), test_texture_bitmap test04/test05 flipped from raises to broadcast asserts; test08_srf_shape_reaches_the_pixel appended to test_specfilm (needs a spectral variant). |
| P5 | path.cpp: budgets + filter-glossy + transparent counter | 1e2aa2a8, 3bf56994, cbfd690b, 48ad0a16, cd96b727, af9fc5e0, dc0a885e, bc42de49, df586f25, faec9175, 38b9f7a3, 58e14ef4, 9314f799, 581004ec (budget removal), 56d2d89a (transparent counter) | tested | Re-expressed by hand onto master's rewritten path.cpp (a291652b churn; upstream now computes si via scene->compute_surface_interaction and threads RayMask::Camera on the first trace). Ported: transparent_max_depth budget + stop_next, clamp_direct/clamp_indirect (clamp_light), blur_glossy filter-glossy floor via si.min_alpha + min_ray_pdf, null-aware MIS bookkeeping guard, null_shadow_transmittance march, per-lobe budget key refusal, Scene::has_null_bsdfs (scene.h/scene.cpp; master's m_shapegroups is ref<>-typed). DEFERRED HUNKS: UV-partials block (reads ray_.diff_scale -> P8); ray-visibility branches + per-lane ray_visibility loop var (-> P6 redesign; shadow march traces RayMask::All until P6 adds the Shadow bit). Tests: 4 fork files copied; 56/57 passed on the cc4f3dca build, and the last (test01, Scene.has_null_bsdfs binding, added at fa41c438) plus the two P6-dependent skips pass on the 192bae19 build -- 76/76 across P5+P6 suites. |
| P6 | Per-object ray visibility (REDESIGN on upstream masks) | 78a88673, d064ec8d, 2eca0cf1, 9770fddb, 50893d38, a1ab146c | tested | REDESIGN per the audit's settled a1d1a1d0 row -- the fork's mechanism (RayVisibility enum + scene-level re-spawn walks ray_test_visible / ray_intersect_preliminary_visible + has_ray_visibility_masks gate) is NOT ported; upstream's RayMask substrate does the filtering in the accel structures. Landed: RayMask extended with Diffuse/Glossy/Transmission/VolumeScatter/Shadow (Camera/All upstream; nonzero-AND matching = Cycles'); Shape parses Blender's visible_* props into m_visibility_mask, combined with the emitter mask in visibility_mask(); InstanceEntry gains its own mask (instance-shape mask AND blas mask -- Blender's switches live on the OBJECT/instance, group data stays All) consumed by OptiX + Metal instance builds; Embree instance geometry now carries the instance's mask; the native kdtree filters top-level shapes via its existing per-prim test (it does not support instancing at all -- scene_native.inl throws, an upstream limitation our embree-disabled build inherits on scalar/llvm); path.cpp carries per-lane ray_visibility (Camera at spawn, per-lobe union after scatters, UNCHANGED across null crossings -- the 4.04x defect), threads it into both traces and the emission ray_mask, shadow march + NEE ray_test carry RayMask::Shadow; RayMask Python enum values added with literal docstrings (docstr regen = P8). Tests: test_ray_visibility.py adapted to the RayMask API (no gate accessors); transparent-boundaries test08/test10 unskipped. |
| P7 | Mesh: blender loader -> from_corners, merge key, CDF determinism | ee7eab61, 4764a6fb, 4f23dea0, 597a9fe8, 80d67542, af95f2f7 | tested | The three blender.cpp loader commits are SUPERSEDED by upstream capabilities, verified on master: ee7eab61 corner normals -> CornerMesh/from_corners is natively corner-indexed with per-corner normals; 4764a6fb zero-normal load tolerance -> upstream 1993fbb9's shading-time revert-to-geometric-normal guard is present in mesh.cpp; 4f23dea0 Generated coords -> CornerMesh.attrs custom vertex attributes (exporter passes them; EXPORTER MIGRATION REQUIRED -- the `blender` shape plugin no longer exists, mitsuba-blender must target Mesh.from_corners). Ported: MergeKey gains visibility_mask + silhouette_sampling_weight (597a9fe8/80d67542 re-expressed on ac04fd30's key; master's merge rebuilds from Properties, so the mask is member-copied and the weight written back -- both defects present on master); merge plugin + Mesh::merge exclude texture-attribute shapes (new Shape::has_texture_attributes); af95f2f7 CDF determinism re-derived onto master's build_pmf (JIT arm migrates areas to host, scalar ctor; scalar variants keep the move ctor). Fork merge tests 03-05 appended to test_merge.py, adapted to RayMask/visibility_mask(). TESTED: the first run FAILED upstream's test_mesh_build test13/test14 on both variants -- the MergeKey addition made upstream's uninitialized m_silhouette_sampling_weight load-bearing (the Shape() default ctor, used by the direct Mesh(name, ...) path, never runs the Properties parse; garbage != garbage, so merge refused every pair). Fixed with an in-class = 1.f initializer (a524be12). After the fix: test_merge 13/13, test_mesh_{build,shading,query} + P5/P6 suites 160 passed, only the 2 known Embree-off kd-tree instancing failures remain. |
| P8 | Ray-diff / bump C++ + remaining tests + docstr regen | cede9dc2, ecb63de2, and the src/python_tests suite | tested | cede9dc2: RayDifferential.diff_scale (ray.h field + ctor inits + scale_differential accumulation + DRJIT_STRUCT entry) applied clean; path.cpp's P5-deferred UV-partials hunk landed (per-pixel footprint at the first vertex, divides by ray_.diff_scale; loop lambda now captures &ray_); bump.py already carried the fork's consumer side via P1. ecb63de2: ray_v.cpp binds diff_scale (literal docstring) + MI_PY_DRJIT_STRUCT gains the field (the silent-loss-under-traversal defect); test_ray_differential_scale.py copied -- the last missing src/python_tests file. TESTED: test_ray_differential_scale passes on the built package (with test_merge, 13/13); P5/P6 + mesh regression sweep clean. docstr.h REGENERATED at f00ad838 (STACK.md recipe; two runs byte-identical, zero fatals; diff is exactly the port's header surface, no identifier removed; the four literal-docstring bindings now use D(...)). Full suite re-run on the final build: 36 failed / 3454 passed / 238 skipped -- failure set identical to the pre-port baseline triage in STACK.md; the one new failure on the first pass (test_interaction test02, exact-repr assert predating min_alpha) fixed by extending the expected string. |
| P9 | Consistent path-space regularization (post-4ded1df8 fork commit) | 0c49cf68 | tested | The one blender-parity feature landed AFTER the audited v3.9.0..4ded1df8 range: Kaplanyan & Dachsbacher 2013 per-sample mollification decayed in roughness space (Weier et al. 2021) on the P2/P5 `si.min_alpha` plumbing. sampler.h (`current_sample_index` protected -> public) and test_regularization.py applied verbatim; the path.cpp hunks re-expressed by hand onto master's rewritten file, where every anchor already existed because P5 re-expressed the blur_glossy/min_ray_pdf machinery they patch (property parse after blur_glossy; per-lane `reg_alpha` from `sampler->current_sample_index()` captured by the recorded-loop lambda; floor-raise after the filter-glossy floor, guarded on `min_ray_pdf` finiteness; min_ray_pdf update guard widened to either feature). TESTED: test_regularization 15/15 (as on blender-parity); path suites 80 passed; samplers 32 passed; full suite 36 failed / 3470 passed / 238 skipped -- failure set identical to the baseline four classes, the +15 passes being this cluster's tests. docstr.h regenerated over the P9 + kd-tree-fix headers (STACK.md recipe; two runs byte-identical; the diff is the P9 `current_sample_index` doc + the 88fd5037 `prim_index` arg, no binding references either). |

Fork commits that are NOT ported, deliberately:

- 067ddf4f, f44a0779, 136eceb3, 9f377688, 62355f7f, 8324a670, 9326b5a2 --
  cherry-picks OF upstream commits; already on master.
- 4ea28b4a (two-sided area) -- superseded by upstream b9286db1
  (audit disposition: drop our arm, re-apply 81ea8f2d on top).
- 26615f7e (test monkeypatch fix) -- folded into P2's test port.
- 581004ec is a REMOVAL (per-lobe budgets); P5 ports the post-removal state,
  so the budgets never appear on this branch.

## P1 adaptations (registration infra + textures)

The fork's mechanism cannot port verbatim; master moved under every piece:

- `alias.cpp` (which carried the fork's per-set_variant reload hunks) was
  deleted upstream (12ef6697); the reload now lives in
  `src/python/__init__.py`'s JIT-variant reload list, which is the same
  mechanism upstream uses for `ad.integrators`/`ad.loaders`. The fork's
  reload-the-submodule-not-the-parent reasoning carries over verbatim.
- `mi.Texture` plugins became `mi.Field` plugins (PR #1885 + master's
  ObjectType::Field registry): new shared base
  `plugins/textures/_base.py::TextureBase` declares the Color3/Surface/JIT
  contract; per-plugin mechanical rewrite (audited, see the port commit):
  `mi.Texture` -> `TextureBase`, `def eval` -> `def eval_color3` (the Field
  dispatch routes spectral eval through out_type to eval_color3; the generic
  `eval` slot now means flat FloatStorage), `register_texture` ->
  `register_field`. eval_1/eval_3/mean/max/resolution/is_spatially_varying/
  traverse/parameters_changed keep their names -- verified against the
  PyField trampoline's ticket table (field_v.cpp).
- **`math` -> `blender_math` rename (BREAKING for exporters):** upstream's
  C++ expression texture `math` (mitsuba3#1944, src/fields/math.cpp) owns
  the name; the registry race is order-dependent, and the C++ plugin even
  shares the `input_0/input_1` property names while requiring `expr` -- the
  RESERVED_BY_CPP guard (839c3f6c) caught the collision live at the first
  smoke run. mitsuba-blender's exporter (`io/exporter/materials.py`, 4
  sites) emits `{'type': 'math', ...}` and must emit `blender_math` against
  this branch.
- RESERVED_BY_CPP refreshed for the field registry: + constvolume,
  gridvolume, math, sinusoidal, volume; the guard regex now matches
  `register_field`.

Evidence: `smoke_p1.py` (session scratchpad) against the built package at
0cba523f6 with the ported Python tree overlaid -- 13/13: registration,
eval_1/eval_3/spectral-eval dispatch, nested-field wrappers (clamp,
color_ramp elements path, texture_coordinate Blender V-flip), C++ `math`
coexistence, variant-change reload re-registration, mean/spatially-varying
passthrough.
