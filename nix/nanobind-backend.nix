# nanobind-backend 1.0.0 -- the COMPILED half of nanobind's split mode.
#
# Both Dr.Jit and Mitsuba build their Python extensions in split mode (their
# CMakeLists default the `*_SPLIT_MODE` options ON, and SKBUILD builds FORCE
# them ON: "Wheels are always built in split mode"). A split-mode extension
# does not statically link libnanobind; at import time it calls
# `nanobind_backend.fill(abi_major, ...)`, which serves the compiled backend
# (`_nb_backend_v1`) out of THIS package. So this is a hard RUNTIME dependency
# of both `drjit.nix` and `mitsuba.nix` -- and a BUILD-time one too, because
# their last ninja step runs nanobind's stubgen, which imports the freshly
# built extension.
#
# Upstream distributes this as binary wheels only ("there is no source
# distribution" -- nanobind-backend/CMakeLists.txt); the project lives INSIDE
# the nanobind repository and builds `src/nb_backend.cpp` from the surrounding
# checkout. Hence `src` is taken from `nanobind.nix`'s checkout (same pinned
# commit, guaranteed by construction) with `sourceRoot` pointed at the
# subproject. Its CMake cross-checks pyproject's version against the ABI
# macros in `include/nanobind/nb_backend.h` and refuses a mislabeled build.
{
  lib,
  buildPythonPackage,
  nanobind3,
  scikit-build-core,
  cmake,
  ninja,
}:

buildPythonPackage rec {
  pname = "nanobind-backend";
  version = "1.0.0";
  pyproject = true;

  # The SAME checkout `nanobind.nix` builds from -- one pin, two packages.
  src = nanobind3.src;
  sourceRoot = "${src.name}/nanobind-backend";

  build-system = [ scikit-build-core ];

  nativeBuildInputs = [
    cmake
    ninja
  ];

  # scikit-build-core drives CMake itself; the cmake setup hook would try to
  # configure a second time in the wrong directory and abort the build.
  dontUseCmakeConfigure = true;

  # Import both the dispatcher package and the compiled v1 backend it serves;
  # upstream's own wheel test-command is the `_nb_backend_v1` import.
  pythonImportsCheck = [
    "nanobind_backend"
    "nanobind_backend._nb_backend_v1"
  ];

  meta = {
    description = "Compiled nanobind backend for extensions built in split mode";
    homepage = "https://github.com/wjakob/nanobind";
    license = lib.licenses.bsd3;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
}
