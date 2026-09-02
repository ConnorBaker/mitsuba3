# hsr/blender-parity -> hsr/master-port: feature-port log

The 84 fork commits (v3.9.0..4ded1df8) ported cluster by cluster onto the
master port stack (see STACK.md). Cluster map and dispositions follow the HSR
repo's `investigations/mitsuba_upstream_audit.md` (§2 delta table, §3
per-commit rows). One row per cluster; update the row when the cluster's
status changes, do not append duplicates.

Statuses: `pending` / `ported` (code landed) / `compiled` / `tested`.

| # | cluster | fork commits | status | notes |
|---|---|---|---|---|
| P1 | Plugin-registration infra + texture plugin tree | d67d0c51 (#1732) + b2b53c47, c604b572, 0b383600, a5f03f7b, 730126e5, eb4150b5, 16522460, be924a8b, 2d130628, 839c3f6c, deda5367, 77cef9d6, adad37b2, 492acb9f/2a557651/7dad744c (bump plugin) | tested (smoke) | See "P1 adaptations" below. 13/13 smoke checks on cuda_ad_rgb incl. variant-change reload; full upstream-suite run pending next compile. |
| P2 | ad/bsdfs (blender_principled + lobes/tables) + BSDF C++ surface | e1bc2905 (#1822), b1fde2f3, 144aaac8, 95c07233, 638d4920, 8b94cda5, a8f6106a, 81fae57d, f1f01f3a, f6c8054f, 596da59a, 991a1844, fcca650e, 3ac0ff8f, 62936c42, b3c64fb7, faec9175 (rough* floors), 77f35439, 4b19696/8b94cda5 tests | tested | C++ (bsdf.h sampled_roughness_squared, interaction.h min_alpha, microfacet.h filter_glossy_alpha, six BSDF cpps, bindings) 3-way applied; 5 conflicts resolved against master's doc-conversion + instance_index/frame_flipped churn. Python tree verbatim + reload entry. principledhelpers.h delta dropped (upstream 366de608 already on master). Compiled at e4d60f01; all 10 suites (44 tests) pass on cuda_ad_rgb -- incl. test_bsdf_sample_struct on the new field, after the pre-compile run failed all 16 min_alpha readers exactly as predicted. |
| P3 | Emitters: area si.wi, spot, blender_lamp | 81ea8f2d (4ea28b4a superseded by upstream b9286db1), c3f60515, 4ed80deb, 4ded1df8, 011aa296 (retired into blender_lamp) | ported | area.cpp: only 81ea8f2d's three si.wi hunks re-applied by hand onto upstream's b9286db1 two-sided implementation (our 4ea28b4a arm dropped per the audit disposition); spot.cpp applied clean 3-way; blender_lamp + emitters registry verbatim + reload entry. test_blender_lamp 12/12 on the P2 binary (python-only); area/spot tests await this cluster's compile. |
| P4 | Sensor/film/rfilter/bitmap | 217be112, f0a48397, f665f9e7 (blackman_harris), a89facc4 + 3d4a9faf (test) | pending | thinlens/orthographic + 6 upstream cherry-picks already ON master: drop. |
| P5 | path.cpp: budgets + filter-glossy + transparent counter | 1e2aa2a8, 3bf56994, cbfd690b, 48ad0a16, cd96b727, af9fc5e0, dc0a885e, bc42de49, df586f25, faec9175, 38b9f7a3, 58e14ef4, 9314f799, 581004ec (budget removal), 56d2d89a (transparent counter) | pending | Master's path.cpp was rewritten (a291652b, a1d1a1d0); port against the audit's settled design. |
| P6 | Per-object ray visibility (REDESIGN on upstream masks) | 78a88673, d064ec8d, 2eca0cf1, 9770fddb, 50893d38, a1ab146c | pending | Audit a1d1a1d0 row: extend upstream RayMask with per-event bits; masks on instances; delete fork re-spawn loops; NEE unchanged. |
| P7 | Mesh: blender loader -> from_corners, merge key, CDF determinism | ee7eab61, 4764a6fb, 4f23dea0, 597a9fe8, 80d67542, af95f2f7 | pending | blender.cpp deleted upstream (9f4c6446); MergeKey gains visibility fields (ac04fd30 row); af95f2f7 must be re-derived against the rewritten mesh.cpp. |
| P8 | Ray-diff / bump C++ + remaining tests + docstr regen | cede9dc2, ecb63de2, and the src/python_tests suite | pending | docstr.h regeneration LAST, with the recorded mkdocs.py recipe in STACK.md. |

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
