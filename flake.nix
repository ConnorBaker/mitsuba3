{
  description = "Mitsuba 3 with Nix packaging: an overlay providing mitsuba, drjit, nanobind 3, nanobind-backend and tinyformat";

  inputs = {
    flake-parts = {
      inputs.nixpkgs-lib.follows = "nixpkgs-lib";
      url = "github:hercules-ci/flake-parts";
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs-lib.url = "github:nix-community/nixpkgs.lib";
    git-hooks-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:cachix/git-hooks.nix";
    };
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };

  # NOTE: this flake MUST be fetched with submodules for the `mitsuba` package
  # to build -- the source it builds is this very tree, and `ext/` is all
  # gitlinks:
  #
  #     nix build '.?submodules=1#mitsuba'
  #     url = "github:ConnorBaker/mitsuba3?ref=<branch>&submodules=1";
  #
  # `nix/mitsuba.nix` asserts this at eval time with a readable error. The
  # other packages (drjit, nanobind3, nanobind-backend, tinyformat) fetch
  # their own pinned sources and build from any checkout.
  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      # The packaging is x86_64-linux-only in practice (CUDA backends); the
      # overlay itself is system-agnostic.
      systems = [ "x86_64-linux" ];

      imports = [
        inputs.treefmt-nix.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];

      flake.overlays.default = import ./nix/overlay.nix;

      perSystem =
        {
          config,
          pkgs,
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [ inputs.self.overlays.default ];
            # cudaPackages.cuda_cudart (a buildInput of drjit and mitsuba) is
            # unfree-redistributable. Consumers applying the overlay configure
            # their own nixpkgs; this only affects this flake's own outputs.
            config.allowUnfree = true;
          };

          pre-commit.settings.hooks = {
            # Formatter checks
            treefmt = {
              enable = true;
              package = config.treefmt.build.wrapper;
            };

            # Nix checks
            deadnix.enable = true;
            nil.enable = true;
            statix.enable = true;
          };

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              # Nix only, deliberately: everything else in this tree (C++,
              # Python, shell, CMake) is upstream Mitsuba source, and
              # reformatting it would poison every future merge with upstream
              # and with the open PR branches stacked on this one.
              nixfmt = {
                enable = true;
                strict = true;
              };
            };
          };

          devShells.default = pkgs.mkShell {
            inputsFrom = [ config.pre-commit.devShell ];
            shellHook = ''
              ${config.pre-commit.installationScript}
            '';
          };

          legacyPackages = pkgs;

          packages = {
            default = pkgs.python3Packages.mitsuba;
            mitsuba = pkgs.python3Packages.mitsuba;
            drjit = pkgs.python3Packages.drjit;
            nanobind3 = pkgs.python3Packages.nanobind3;
            nanobind-backend = pkgs.python3Packages.nanobind-backend;
            inherit (pkgs) tinyformat;
          };
        };
    };
}
