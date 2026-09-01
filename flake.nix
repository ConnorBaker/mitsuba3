{
  description = "Mitsuba 3 with Nix packaging: an overlay providing mitsuba, drjit, nanobind 3, nanobind-backend and tinyformat";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  # NOTE: this flake MUST be fetched with submodules for the `mitsuba` package
  # to build -- the source it builds is `self`, and `ext/` is all gitlinks:
  #
  #     nix build '.?submodules=1#mitsuba'
  #     url = "github:ConnorBaker/mitsuba3?ref=<branch>&submodules=1";
  #
  # `nix/mitsuba.nix` asserts this at eval time with a readable error. The
  # other packages (drjit, nanobind3, nanobind-backend, tinyformat) fetch
  # their own pinned sources and build from any checkout.
  outputs =
    { self, nixpkgs }:
    let
      # The packaging is x86_64-linux-only in practice (CUDA backends); the
      # overlay itself is system-agnostic.
      system = "x86_64-linux";
      overlay = import ./nix/overlay.nix { mitsubaSrc = self; };
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ overlay ];
        # cudaPackages.cuda_cudart (a buildInput of drjit and mitsuba) is
        # unfree-redistributable. Consumers applying the overlay configure
        # their own nixpkgs; this only affects this flake's `packages` output.
        config.allowUnfree = true;
      };
      ps = pkgs.python3Packages;
    in
    {
      overlays.default = overlay;

      packages.${system} = {
        default = ps.mitsuba;
        mitsuba = ps.mitsuba;
        drjit = ps.drjit;
        nanobind3 = ps.nanobind3;
        nanobind-backend = ps.nanobind-backend;
        tinyformat = pkgs.tinyformat;
      };
    };
}
