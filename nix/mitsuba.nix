# Mitsuba 3, BUILT FROM SOURCE -- from THIS repository's own tree.
#
# WHERE THE SOURCE COMES FROM. `src` is the `../.` path -- the checkout this
# file lives in, with no argument threading. THE TREE MUST INCLUDE THE
# SUBMODULES: a flake fetched without `?submodules=1` is missing `ext/`, and
# the top-level CMakeLists refuses to configure ("The Mitsuba 3 dependencies
# are missing!"). The eval-time assert below turns that into a readable error
# instead of a mid-build one.
#
# SPLIT MODE: SKBUILD forces `MI_SPLIT_MODE=ON` ("Wheels are always built in
# split mode"), so the Python extension delegates to the compiled nanobind
# backend at import time -- `nanobind-backend` is a hard runtime dependency,
# and a build-time one (stubgen imports the fresh extension). Nanobind itself
# comes from `nanobind.nix` via `python -m nanobind --cmake_dir`, not from the
# `ext/nanobind` submodule; both are the same pinned commit.
#
# DR.JIT IS NOT BUILT TWICE. `ext/CMakeLists.txt` guards its vendored Dr.Jit
# with `if (NOT SKBUILD)`, and scikit-build-core sets SKBUILD -- so this build
# runs `python -c "import drjit; print(drjit.get_cmake_dir())"` and then
# `find_package(drjit CONFIG REQUIRED)`, linking the SAME `drjit` derivation
# that `drjit.nix` produces. The submodule tree is still fetched and still
# required: the top-level `CMakeLists.txt` refuses to configure without
# `ext/drjit/ext/drjit-core/ext/nanothread/ext/cmake-defaults`, whose
# CMakeLists it includes for the project's compile defaults.
#
# THE VARIANT SET IS DELIBERATE. `scalar_rgb` is not optional -- Mitsuba
# hard-requires it as the variant its plugin registry and `mitsuba.scalar_rgb`
# bootstrap path are written against (and a `scalar_rgb`-ONLY set does not even
# LINK: `jit_free`/`jit_flag` go undefined from the `Mesh.from_corners` port and
# `SurfaceField::eval_1`, so at least one llvm/cuda variant is mandatory too).
#
# The two `_polarized` names were added 2026-09-05 for the consuming project's
# LENS model, and the reason is not polarization for its own sake: the element
# trace's own docstring lists `reflections, coatings` as absent-not-approximated,
# and per-surface Fresnel transmission and an AR stack split into s- and p-
# components, so they cannot be added correctly on unpolarized transport. The
# colour axis and the polarization axis are ORTHOGONAL -- spectral buys the
# wavelength physics (diffraction, dispersion, TCA), polarized buys the interface
# physics; neither substitutes for the other.
#
# `scalar_spectral_polarized` earns its place separately: it is the only CPU
# spectral variant, and spectral-side verification (e.g. the SRF `pdf_spectrum`
# probes) otherwise has to occupy the exclusive GPU to run at all.
#
# COST, MEASURED rather than assumed -- the previous text here claimed every
# extra variant is "a full recompile of the entire plugin tree", which
# over-priced it. Five variants build in ~375 s wall against ~103 s for the
# three below, i.e. roughly 136 s marginal per POLARIZED variant (~3.4x the
# ~40 s an unpolarized one costs), farmed to the remote builder. Minutes, not
# hours. Override `mitsubaVariants` for a different set.
{
  lib,
  buildPythonPackage,
  autoAddDriverRunpath,
  cmake,
  ninja,
  nasm,
  scikit-build-core,
  nanobind3,
  nanobind-backend,
  drjit,
  numpy,
  typing-extensions,
  hatch-fancy-pypi-readme,
  pathspec,
  pyproject-metadata,
  cudaPackages,
  zlib,
  pugixml,
  fast-float,
  tinyformat,
  python,
  stdenv,
  mitsubaVariants ? [
    "scalar_rgb"
    "cuda_ad_rgb"
    "cuda_ad_spectral"
    "scalar_spectral_polarized"
    "cuda_ad_spectral_polarized"
  ],
  # EXPERIMENTAL, default OFF (see the MI_ENABLE_EMBREE note at `cmakeFlags`
  # for why off is the deliberate default). ON builds the vendored
  # `ext/embree` (internal tasking system, no extra Nix dependency) as the
  # scalar-variant CPU ray-tracing backend in place of Mitsuba's native
  # kd-tree. Exposed as the `mitsuba-embree` flake package for measuring the
  # kd-tree-vs-Embree test-suite delta; flipping the DEFAULT is a decision
  # for the consuming project, not this file.
  enableEmbree ? false,
}:

let
  src = ../.;

  # CARRY THE VENDORED TREE'S ONE-LINE SERIALIZATION PATCH onto the stock
  # pugixml this build links. Upstream's `ext/pugixml` is not pristine: in
  # `node_output_start` it deletes the two lines that write pugixml's
  # traditional space before an empty element's closing tag ("wenzel: removed
  # this extra space before the closing tag"), so vendored builds emit `/>`
  # where stock pugixml emits ` />`. Three upstream tests assert the `/>`
  # form byte-for-byte (test_parser_writing::test06/test07,
  # test_field::test01), and every mitsuba-written XML file changes shape
  # without it -- the patch is behavior, not style, so unvendoring must
  # preserve it. Scoped HERE rather than in overlay.nix: the overlay
  # deliberately adds only new attribute names, and this must not leak a
  # patched pugixml into consumers' unrelated packages.
  pugixml' = pugixml.overrideAttrs (prevAttrs: {
    patches = (prevAttrs.patches or [ ]) ++ [ ./pugixml-empty-tag-no-space.patch ];
  });

  # The variant grammar, straight from `resources/mitsuba.conf.template`:
  # backend, then the optional `ad` feature (JIT backends only -- the template
  # defines no scalar_ad_* variants), then a color representation, then the
  # optional `polarized` and `double` features, in that order. This
  # enumeration is exactly the 60 variant names the template defines;
  # `MI_DEFAULT_VARIANTS` can only select among defined names, so anything
  # outside this list would fail the build much later with a far worse error.
  validVariants =
    lib.concatMap
      (
        backend:
        lib.concatMap (
          ad:
          lib.concatMap
            (
              color:
              lib.concatMap
                (
                  pol:
                  map (dbl: "${backend}${ad}_${color}${pol}${dbl}") [
                    ""
                    "_double"
                  ]
                )
                [
                  ""
                  "_polarized"
                ]
            )
            [
              "mono"
              "rgb"
              "spectral"
            ]
        ) ([ "" ] ++ lib.optional (backend != "scalar") "_ad")
      )
      [
        "scalar"
        "llvm"
        "cuda"
      ];

  unknownVariants = lib.subtractLists validVariants mitsubaVariants;
in

# The readable form of the missing-submodules failure -- see the note at the
# top of this file. Checked at eval so `nix build .#mitsuba` (without
# `?submodules=1`) fails in milliseconds with the fix in the message, instead
# of after a source copy with CMake's generic complaint.
assert lib.assertMsg
  (builtins.pathExists (
    src + "/ext/drjit/ext/drjit-core/ext/nanothread/ext/cmake-defaults/CMakeLists.txt"
  ))
  ''
    The mitsuba3 source tree is missing its submodules. Fetch the flake with
    submodules enabled, e.g.:  nix build '.?submodules=1#mitsuba'  or, as an
    input:  url = "github:ConnorBaker/mitsuba3?ref=<branch>&submodules=1";
  '';
assert lib.assertMsg (unknownVariants == [ ]) ''
  mitsubaVariants contains unknown variant name(s): ${lib.concatStringsSep ", " unknownVariants}
  A variant is <backend>[_ad]_<color>[_polarized][_double] with backend one of
  scalar/llvm/cuda (ad requires llvm or cuda) and color one of
  mono/rgb/spectral -- see resources/mitsuba.conf.template.
'';
assert lib.assertMsg (lib.elem "scalar_rgb" mitsubaVariants) ''
  mitsubaVariants must include "scalar_rgb": Mitsuba's plugin registry and
  `mitsuba.scalar_rgb` bootstrap path hard-require it
  (resources/mitsuba.conf.template: 'the "scalar_rgb" variant *must* be
  included at the moment').
'';

buildPythonPackage (finalAttrs: {
  pname = "mitsuba";
  # MUST MATCH the version the built wheel declares, which scikit-build-core
  # reads from `include/mitsuba/mitsuba.h` (MI_VERSION_* + MI_VERSION_DEV).
  # `pythonMetadataCheckPhase` parses both with `packaging.version.Version`
  # and fails on any mismatch. Do not "fix" a failure by tuning this string
  # until the check goes quiet unless the header actually moved.
  version = "3.10.0.dev1";
  pyproject = true;

  inherit src;

  build-system = [
    scikit-build-core
    nanobind3
    drjit
    typing-extensions
    hatch-fancy-pypi-readme
    pathspec
    pyproject-metadata
  ];

  nativeBuildInputs = [
    cmake
    ninja
    # libjpeg-turbo's SIMD paths are NASM; `ext/CMakeLists.txt` probes for it
    # with `check_language(ASM_NASM)` and silently drops to the C fallback
    # when absent.
    nasm
    # Must run last: it re-adds the driver runpath that other fixups strip.
    autoAddDriverRunpath
  ];

  # scikit-build-core drives CMake itself; the cmake setup hook would try to
  # configure a second time in the wrong directory and abort the build.
  dontUseCmakeConfigure = true;

  buildInputs = [
    cudaPackages.cuda_cudart
    # ZLIB IS ALREADY UNVENDORED, and not by choice on our part --
    # `ext/CMakeLists.txt` only builds the bundled `ext/zlib` under
    # `if (WIN32)`, so on Linux the vendored OpenEXR's `OpenEXRSetup.cmake`
    # does a plain `find_package(ZLIB)` and hard-fails ("Could NOT find ZLIB")
    # against nothing. nixpkgs' zlib satisfies it directly.
    zlib
    # libatomic, for the same nanothread compare-and-swap reason documented in
    # drjit.nix -- the vendored nanothread is reachable from this build too.
    (lib.getLib stdenv.cc.cc)
    # UNVENDORED out of `ext/`, see postPatch. nixpkgs' pugixml ships only
    # `lib/libpugixml.a`, so the result is a STATIC link -- `libmitsuba.so`
    # grows and no `libpugixml.so` is installed beside it, where the vendored
    # build shipped one. fast-float and tinyformat are header-only and
    # contribute include paths only. `pugixml'` (not the stock attr) carries
    # the vendored tree's serialization patch -- see the note at `pugixml'`.
    pugixml'
    fast-float
    tinyformat
  ];

  dependencies = [
    drjit
    # Split mode -- see the note at the top of this file.
    nanobind-backend
    # Several `mitsuba.python` utility modules import numpy at module scope,
    # and the build's own stubgen step imports the package.
    numpy
  ]
  ++ lib.optionals (python.pythonOlder "3.11") [ typing-extensions ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib")
    (lib.cmakeFeature "MI_DEFAULT_VARIANTS" (lib.concatStringsSep "," mitsubaVariants))
    # Embree is the CPU/BVH ray-tracing backend. The default variants are
    # CUDA-only (OptiX supplies the acceleration structure there) and
    # `scalar_rgb` falls back to Mitsuba's own kd-tree, so building Embree's
    # full ISA matrix -- SSE42/AVX/AVX2/AVX512SKX, each a separate compile of
    # the whole kernel set -- buys nothing here. `enableEmbree` (above) flips
    # this for the experimental Embree-on measurement build.
    (lib.cmakeBool "MI_ENABLE_EMBREE" enableEmbree)
    (lib.cmakeBool "MI_ENABLE_PYTHON" true)
    # Bundled deps (ext/zlib among them) declare `cmake_minimum_required`
    # values that CMake >= 4.0 rejects outright. Upstream sets this in
    # pyproject.toml for the same reason; it is repeated here because
    # cmakeFlags and scikit-build's own defines are merged, not one overriding
    # the other.
    (lib.cmakeFeature "CMAKE_POLICY_VERSION_MINIMUM" "3.5")
  ];

  env.NIX_CFLAGS_COMPILE = "-Wno-error";

  # UNVENDOR THE THREE `ext/` TREES THAT CAN BE UNVENDORED, and only those
  # three:
  #
  #   pugixml   -> nixpkgs pugixml
  #   fastfloat -> nixpkgs fast-float
  #   tinyformat-> ./tinyformat.nix (the SAME commit `ext/tinyformat` pins; a
  #                pure relocation)
  #
  # THE OTHERS ARE NOT OVERSIGHTS, and the reasons are in the files being
  # patched:
  #
  #   openexr / libpng / libjpeg-turbo -- upstream renames every one of these
  #     targets (`OUTPUT_NAME "<X>-mitsuba"`) under the comment "Give
  #     libpng & libjpeg a name that's guaranteeed not to match other
  #     libraries that may already be loaded (e.g. into a Python
  #     interpreter)". That is exactly the consuming project's situation --
  #     mitsuba is imported alongside torch, OpenCV and imageio, all of which
  #     bring their own libpng/libjpeg/OpenEXR. Unvendoring these removes a
  #     deliberate collision guard, so they stay.
  #   struct-jit / rgb2spec -- mitsuba-renderer's own, not packaged anywhere.
  #   embree -- not built by default (MI_ENABLE_EMBREE defaults off via
  #     `enableEmbree`); when enabled it builds vendored, as upstream does.
  #   zlib -- already unvendored; `ext/zlib` is built only `if (WIN32)`.
  #   nanobind, drjit -- already come from Nix (see the top note).
  #
  # `mitsuba-core` links the literal target names `pugixml` and `fast_float`,
  # so each replacement keeps that name as an ALIAS of the imported target
  # rather than editing every consumer.
  postPatch = ''
        substituteInPlace ext/CMakeLists.txt \
          --replace-fail \
            'add_library(pugixml SHARED pugixml/src/pugixml.cpp)' \
            'find_package(pugixml REQUIRED)
        add_library(pugixml_shim INTERFACE)
        target_link_libraries(pugixml_shim INTERFACE pugixml::pugixml)' \
          --replace-fail \
            'set(PUGIXML_INCLUDE_DIRS ''${CMAKE_CURRENT_SOURCE_DIR}/pugixml/src PARENT_SCOPE)' \
            'set(PUGIXML_INCLUDE_DIRS ${lib.getDev pugixml'}/include PARENT_SCOPE)' \
          --replace-fail \
            'set(TINYFORMAT_INCLUDE_DIRS ''${CMAKE_CURRENT_SOURCE_DIR}/tinyformat PARENT_SCOPE)' \
            'set(TINYFORMAT_INCLUDE_DIRS ${tinyformat}/include PARENT_SCOPE)' \
          --replace-fail \
            'add_subdirectory(fastfloat EXCLUDE_FROM_ALL)' \
            'find_package(FastFloat REQUIRED)
        add_library(fast_float INTERFACE IMPORTED GLOBAL)
        target_link_libraries(fast_float INTERFACE FastFloat::fast_float)'

        # The `set_property(TARGET pugixml ...)` lines and LIBRARY_OUTPUT_DIRECTORY
        # only make sense for a target this build owns; with pugixml imported they
        # would fail ("is not built by this project"), so drop them and point
        # consumers at the shim.
        substituteInPlace ext/CMakeLists.txt \
          --replace-fail 'set_property(TARGET pugixml PROPERTY
      LIBRARY_OUTPUT_DIRECTORY "''${CMAKE_CURRENT_BINARY_DIR}/pugixml")
    set_property(TARGET pugixml PROPERTY FOLDER "dependencies")' "" \
          --replace-fail 'target_compile_options(pugixml PRIVATE -DPUGIXML_BUILD_DLL)
    target_compile_features(pugixml PUBLIC cxx_std_17)' ""

        substituteInPlace src/core/CMakeLists.txt \
          --replace-fail '  pugixml' '  pugixml_shim'

        # `MI_DEPEND` is the list of dependency libraries the wheel COPIES IN and
        # installs beside `libmitsuba.so`. A pugixml that comes from the Nix store
        # is linked, not vendored, so it must leave that list -- otherwise
        # `set_target_properties` and `install(TARGETS ...)` both fail on a target
        # this project no longer builds. This is the substantive half of the
        # unvendoring: it is what stops `libpugixml.so` shipping inside
        # `site-packages/mitsuba/`.
        substituteInPlace CMakeLists.txt \
          --replace-fail 'IlmImf IlmThread Imath Iex IexMath Half pugixml' \
                         'IlmImf IlmThread Imath Iex IexMath Half'
  '';

  # POINT THE DR.JIT CHECK AT THE DR.JIT WE ACTUALLY BUILT AGAINST, rather
  # than deleting it. Upstream's `__init__.py` infers the expected Dr.Jit from
  # a SIBLING directory (`<mitsuba>/../drjit`), which is a fine heuristic for
  # a venv and simply false under Nix: the realpath resolves the env symlink
  # back into Mitsuba's own store output, whose parent contains no `drjit`, so
  # the check warns on every import even though the two are the same build.
  # Substituting the concrete store path keeps the guard and makes it
  # STRICTER: it now names one specific build, so a foreign or rebuilt Dr.Jit
  # still trips it.
  postInstall = ''
    substituteInPlace $out/${python.sitePackages}/mitsuba/__init__.py \
      --replace-fail \
        "_drjit_expected_loc = _os.path.realpath(_os.path.join(__path__[0], '..', 'drjit'))" \
        '_drjit_expected_loc = _os.path.realpath("${drjit}/${python.sitePackages}/drjit")'
  '';

  # Dr.Jit initializes a cache under $HOME the moment it is imported, and
  # Mitsuba's build imports it twice: once for `drjit.get_cmake_dir()` during
  # configure, and again when nanobind's stubgen loads the freshly built
  # extension. Nix's default HOME does not exist, so both would abort. See the
  # longer note in drjit.nix.
  preBuild = ''
    export HOME=$(mktemp -d)
  '';

  pythonImportsCheck = [ finalAttrs.pname ];

  meta = {
    description = "Mitsuba 3: a retargetable forward and inverse renderer";
    homepage = "https://github.com/mitsuba-renderer/mitsuba3";
    license = lib.licenses.bsd3;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
})
