# Dr.Jit 1.6.0.dev1, BUILT FROM SOURCE, pinned to the EXACT commit this
# repository's `ext/drjit` gitlink carries -- if that submodule moves, move
# `rev` (and the hash) with it. `pyproject.toml` at the repo root pins
# `drjit==1.6.0.dev1`, and the runtime dependency check compares that against
# the version the built wheel declares (read from `include/drjit/fwd.h`), so
# the two must stay in lockstep.
#
# SPLIT MODE: this build produces a split-mode extension (SKBUILD forces
# `DRJIT_SPLIT_MODE=ON`; "Wheels are always built in split mode"). The
# extension delegates to the compiled nanobind backend at import time, which
# makes `nanobind-backend` a hard runtime AND build-time dependency (the
# build's own stubgen step imports the fresh extension). See
# `nanobind-backend.nix`.
#
# NANOBIND COMES FROM `nanobind.nix`, NOT FROM THE SUBMODULE. Under SKBUILD,
# `ext/cmake-defaults` locates nanobind by running `python -m nanobind
# --cmake_dir` -- the Python package is the source of truth and the vendored
# `ext/nanobind` is ignored on that path. `nanobind.nix` packages the same
# commit the submodule points at, so nothing can disagree.
#
# SUBMODULES ARE FETCHED, NOT UNVENDORED, and that is a decision rather than
# laziness -- see `mitsuba.nix` for the ones that ARE unvendored. Dr.Jit
# vendors:
#
#   ext/drjit-core                -- upstream's own, no separate release exists
#   ext/drjit-core/ext/nanothread -- likewise
#   ext/drjit-core/ext/robin_map  -- `mitsuba-renderer/robin-map`, and this one
#                                    CANNOT be unvendored. The fork adds
#                                    `insert_hash` / `insert_or_assign_hash` /
#                                    `try_emplace_hash` on top of Tessil's
#                                    upstream, and `drjit-core/src/var.cpp`
#                                    calls `try_emplace_hash` in the
#                                    local-value-numbering path. nixpkgs ships
#                                    the same-numbered version WITHOUT those
#                                    commits, so substituting it does not fail
#                                    at eval -- it fails to compile, and the
#                                    tempting repair (rewrite the call as
#                                    `try_emplace`) silently deletes a
#                                    hash-reuse fast path.
#   ext/nanobind                  -- IGNORED at build time; see above.
{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  autoAddDriverRunpath,
  cmake,
  ninja,
  scikit-build-core,
  nanobind3,
  nanobind-backend,
  typing-extensions,
  hatch-fancy-pypi-readme,
  pathspec,
  pyproject-metadata,
  llvmPackages_20,
  cudaPackages,
  python,
  stdenv,
}:

buildPythonPackage rec {
  pname = "drjit";
  version = "1.6.0.dev1";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "mitsuba-renderer";
    repo = "drjit";
    # The commit `ext/drjit` points at in this repository (mitsuba3 master's
    # pin); an ancestor of drjit master, not a release tag.
    rev = "de73a6a2db3c860903899f4dbf7169d11106ea6c";
    fetchSubmodules = true;
    hash = "sha256-3tFJIUy+JdCaF+AZ3Qx9aRlnD6trfZ/Xsr5hqHyTyvc=";
  };

  build-system = [
    scikit-build-core
    nanobind3
    typing-extensions
    hatch-fancy-pypi-readme
    pathspec
    pyproject-metadata
  ];

  nativeBuildInputs = [
    cmake
    ninja
    # Must run last: it re-adds the driver runpath that other fixups strip.
    autoAddDriverRunpath
  ];

  # scikit-build-core drives CMake itself; the cmake setup hook would try to
  # configure a second time in the wrong directory and abort the build.
  dontUseCmakeConfigure = true;

  buildInputs = [
    llvmPackages_20.llvm.lib
    cudaPackages.cuda_cudart
    # libatomic, for nanothread's 16-byte compare-and-swap. GCC does not link
    # it implicitly, and `ext/drjit-core/ext/nanothread/CMakeLists.txt`
    # hard-fails when `find_library(LIBATOMIC ...)` comes up empty -- which it
    # does here, because that call passes no HINTS and libatomic lives in gcc's
    # own `lib` output rather than in any default prefix. CMAKE_LIBRARY_PATH
    # below is what actually makes it findable; this entry is what puts the
    # runtime dependency in the closure.
    (lib.getLib stdenv.cc.cc)
  ];

  dependencies = [
    # Split mode: the extension is served by the compiled backend at import
    # time (see the note at the top of this file).
    nanobind-backend
  ]
  ++ lib.optionals (python.pythonOlder "3.11") [ typing-extensions ];

  # NO OPTIX SDK IS REQUIRED, which is worth stating because it is the
  # opposite of the usual expectation for an OptiX-capable build.
  # `drjit-core/src/optix_api.cpp` dlopens `libnvoptix.so.1` from the driver
  # and re-declares the whole OptiX ABI itself (`using OptixDeviceContext =
  # void*;` plus a table of `DR_OPTIX_SYM` function pointers); every
  # `#include "optix.h"` in that tree is quoted, i.e. its own header. So OptiX
  # is a RUNTIME dependency satisfied by `/run/opengl-driver/lib`, and
  # `autoAddDriverRunpath` above is what makes it resolvable.
  cmakeFlags = [
    # See the libatomic note in buildInputs: nanothread's bare `find_library`
    # searches only CMake's default prefixes, so the path has to be handed to
    # it explicitly.
    (lib.cmakeFeature "CMAKE_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib")
    (lib.cmakeBool "DRJIT_ENABLE_LLVM" true)
    (lib.cmakeBool "DRJIT_ENABLE_CUDA" true)
    (lib.cmakeBool "DRJIT_ENABLE_AUTODIFF" true)
    (lib.cmakeBool "DRJIT_ENABLE_PYTHON" true)
    # Compiling the Dr.Jit test suite is documented upstream as taking "*very*
    # long"; the import smoke below plus the consumers' own suites are the
    # checks relied on.
    (lib.cmakeBool "DRJIT_ENABLE_TESTS" false)
  ];

  env.NIX_CFLAGS_COMPILE = "-Wno-error";

  # A WRITABLE HOME IS A BUILD-TIME REQUIREMENT, not just a test-time one. The
  # last ninja step runs nanobind's `stubgen.py`, which IMPORTS the freshly
  # built extension to generate the `.pyi` stubs; that import calls
  # `jit_init()`, which creates a cache at `$HOME/.drjit` and aborts the whole
  # build when it cannot ("creation of directory /homeless-shelter/.drjit
  # failed"). Nix's default HOME does not exist, so every phase from
  # buildPhase onward needs a real one.
  preBuild = ''
    export HOME=$(mktemp -d)
  '';

  # MAKE THE LLVM BACKEND ACTUALLY RESOLVE. `drjit-core/src/llvm_api.cpp`
  # dlopens the bare soname "libLLVM.so" (falling back to a glob under
  # /usr/lib, which does not exist here), so listing LLVM in buildInputs is
  # not enough -- it satisfies the build and the backend then reports itself
  # ABSENT at runtime. Measured before this line existed:
  # `dr.has_backend(dr.JitBackend.LLVM)` returned 0 on a build that had
  # advertised "Dr.Jit: building the LLVM backend", which is precisely the
  # advertised-but-unenforced shape worth refusing to ship. nixpkgs does
  # provide the bare `libLLVM.so` symlink, so a runpath entry is all that is
  # needed; the DRJIT_LIBLLVM_PATH env override exists too, but a package that
  # works only when the caller exports a variable is a trap for the next
  # person.
  postFixup = ''
    patchelf --add-rpath ${lib.getLib llvmPackages_20.llvm}/lib \
      "$out/${python.sitePackages}/drjit/libdrjit-core.so"
  '';

  pythonImportsCheck = [ "drjit" ];

  meta = {
    description = "Just-in-time compiler for differentiable rendering";
    homepage = "https://github.com/mitsuba-renderer/drjit";
    license = lib.licenses.bsd3;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
}
