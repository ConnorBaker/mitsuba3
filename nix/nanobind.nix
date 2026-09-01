# nanobind 3.0.0, pinned to the EXACT commit this repository's submodules carry.
#
# WHY THIS IS PACKAGED HERE rather than taken from nixpkgs: under SKBUILD, both
# Dr.Jit's and Mitsuba's CMake (via `ext/cmake-defaults`) locate nanobind by
# running `python -m nanobind --cmake_dir` and then `find_package(nanobind
# CONFIG REQUIRED)` -- i.e. the PYTHON PACKAGE is the source of truth for the
# headers and CMake machinery, and the vendored `ext/nanobind` submodule is
# IGNORED on that path. `pyproject.toml` pins `nanobind==3.0.0`; packaging the
# same commit the submodules point at (tag `backend-v1.0.0`, which is the
# "v3.0.0 release" commit) makes the package and the gitlinks byte-identical,
# so there is no version to reconcile.
#
# The rev is what `ext/nanobind` (and `ext/drjit/ext/nanobind`) resolve to at
# this repository's HEAD -- if those gitlinks move, move this with them.
#
# This is a headers+CMake package (`wheel.platlib = false`): nothing is
# compiled into it. The COMPILED half of split mode is `nanobind-backend.nix`,
# built from the same checkout.
{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  scikit-build-core,
  cmake,
  ninja,
}:

buildPythonPackage rec {
  pname = "nanobind";
  version = "3.0.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "wjakob";
    repo = "nanobind";
    # Tag `backend-v1.0.0` dereferenced to its commit ("v3.0.0 release").
    rev = "1bd3139721d6176dfb10979b7219ccb58ef8aafa";
    # ext/robin_map -- nanobind's internal hash table, required by consumers'
    # builds through the CMake config's include paths.
    fetchSubmodules = true;
    hash = "sha256-Rx+ZJRpsUwrTTqLva38cT/rp0QmHOo+9CEEouSWZ4NU=";
  };

  build-system = [ scikit-build-core ];

  nativeBuildInputs = [
    cmake
    ninja
  ];

  # scikit-build-core drives CMake itself; the cmake setup hook would try to
  # configure a second time in the wrong directory and abort the build.
  dontUseCmakeConfigure = true;

  pythonImportsCheck = [ "nanobind" ];

  meta = {
    description = "Tiny and efficient C++/Python bindings";
    homepage = "https://github.com/wjakob/nanobind";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.all;
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
}
