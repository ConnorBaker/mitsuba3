# The overlay: Mitsuba 3 and its dependency stack, added to nixpkgs.
#
# Python packages go through `pythonPackagesExtensions` -- the one mechanism
# that reaches EVERY interpreter's package set -- so consumers can name them
# off `python3Packages` (or any `pythonXYPackages`) like any other dependency.
# All additions are NEW attribute names (`nanobind3`, `nanobind-backend`,
# `drjit`, `mitsuba`, and the top-level `tinyformat`), so no derivation that
# already builds in a consumer's package set changes because of this overlay.
# In particular the set's existing `nanobind` attribute is left alone:
# `nanobind3` is scoped here because Dr.Jit and Mitsuba pin nanobind 3.0.0
# while consumers may rely on whatever their nixpkgs pins.
#
# ONE ARGUMENT, and it is a SOURCE, not a package. `mitsubaSrc` is the
# mitsuba3 source tree -- `flake.nix` passes its own `self`, and it MUST have
# been fetched with submodules (`?submodules=1`); `mitsuba.nix` asserts this
# at eval time with the fix in the message.
{ mitsubaSrc }:
_final: prev: {
  # NOT a Python package: a C++ header library that `mitsuba.nix` unvendors
  # out of `ext/tinyformat`. nixpkgs has no `tinyformat` attribute, so the
  # alternative to packaging it is leaving it vendored.
  tinyformat = prev.callPackage ./tinyformat.nix { };

  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
    (pythonFinal: _pythonPrev: {
      nanobind3 = pythonFinal.callPackage ./nanobind.nix { };
      nanobind-backend = pythonFinal.callPackage ./nanobind-backend.nix { };
      drjit = pythonFinal.callPackage ./drjit.nix { };
      mitsuba = pythonFinal.callPackage ./mitsuba.nix { src = mitsubaSrc; };
    })
  ];
}
