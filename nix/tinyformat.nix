# tinyformat -- a single-header type-safe printf, unvendored out of `mitsuba3/ext`.
#
# THIS IS NOT A PYTHON PACKAGE. It lives here because it exists only to serve
# `mitsuba.nix`, which is; `overlay.nix` adds it at the TOP level of pkgs rather than
# into a Python set. nixpkgs has no `tinyformat` attribute (checked), so unvendoring it
# means packaging it, not just naming it.
#
# PINNED TO MITSUBA'S OWN SUBMODULE REVISION rather than to an upstream tag. tinyformat
# has no releases after 2.3.0 and mitsuba tracks a commit; taking the same commit makes
# this an unvendoring rather than a version bump, so the only thing that changed is WHERE
# the header comes from. The rev is what `ext/tinyformat` resolves to at the mitsuba3 rev
# in `mitsuba.nix` -- if that moves, re-read it and move this with it.
{
  lib,
  stdenvNoCC,
  fetchFromGitHub,
}:

stdenvNoCC.mkDerivation {
  pname = "tinyformat";
  version = "2.3.0-unstable-2019-12-04";

  src = fetchFromGitHub {
    owner = "c42f";
    repo = "tinyformat";
    rev = "635345c75bd95891ee041ac51ce74ebc891d5bab";
    hash = "sha256-4kegaAMxFx3mOZTUj6+HC9D5Lx2e3hcTw85dmDnmlRs=";
  };

  # The whole library is one header. Upstream's CMakeLists builds only its test binary,
  # so there is nothing to configure or compile -- installing the header IS the package.
  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    install -Dm444 tinyformat.h "$out/include/tinyformat.h"
    runHook postInstall
  '';

  meta = {
    description = "Minimal type-safe printf replacement library for C++";
    homepage = "https://github.com/c42f/tinyformat";
    license = lib.licenses.boost;
    platforms = lib.platforms.all;
    sourceProvenance = with lib.sourceTypes; [ fromSource ];
  };
}
