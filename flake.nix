{
  description = "Quantifying performance of machine-learning optimizers like RMSProp & Adam.";
  inputs = {
    check-and-compile = {
      inputs = {
        flake-utils.follows = "flake-utils";
        nixpkgs.follows = "nixpkgs";
      };
      url = "github:wrsturgeon/check-and-compile";
    };
    flake-utils.url = "github:numtide/flake-utils";
    nixfmt = {
      inputs.flake-utils.follows = "flake-utils";
      url = "github:serokell/nixfmt";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      check-and-compile,
      flake-utils,
      nixfmt,
      nixpkgs,
      self,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pname = "meta-optimizer";
        version = "0.0.1";
        src = ./.;
        pkgs = import nixpkgs { inherit system; };
        pypkgs = pkgs.python311Packages;
        # TODO: Use pylyzer when 1.76.0+ supported
        default-pkgs =
          p: py:
          with py;
          [
            beartype
            jaxtyping
            matplotlib
          ]
          ++ [
            (check-and-compile.lib.with-pkgs p py)
            (jax.overridePythonAttrs (
              old:
              old
              // {
                doCheck = false;
                propagatedBuildInputs = old.propagatedBuildInputs ++ [ py.jaxlib-bin ];
              }
            ))
          ];
        check-pkgs =
          p: py: with py; [
            hypothesis
            pytest
          ];
        ci-pkgs =
          p: py: with py; [
            black
            coverage
          ];
        dev-pkgs = p: py: with py; [ python-lsp-server ];
        lookup-pkg-sets =
          ps: p: py:
          builtins.concatMap (f: f p py) ps;
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps pkgs)}/bin/python";
        instantiate-default = s: if s == "default" then pname else s;
        apps = {
          ci =
            let
              find = "${pkgs.findutils}/bin/find";
              nixfmt-bin = "${nixfmt.packages.${system}.default}/bin/nixfmt";
              python = python-with [
                default-pkgs
                check-pkgs
                ci-pkgs
              ];
              rm = "${pkgs.coreutils}/bin/rm";
              xargs = "${pkgs.findutils}/bin/xargs";
            in
            ''
              ${rm} -fr result
              ${find} . -name '*.nix' | ${xargs} ${nixfmt-bin} --check
              ${python} -m black --check .
              ${python} -m coverage run --omit='/nix/*' -m pytest -Werror test.py
              ${python} -m coverage report -m --fail-under=100
            '';
          default = ''
            ${python-with [ default-pkgs ]} $out/main.py
          '';
          plot = ''
            ${python-with [ default-pkgs ]} $out/plot.py
          '';
          plot-convergence = ''
            ${python-with [ default-pkgs ]} $out/plot-convergence.py
          '';
        };
      in
      {
        apps = builtins.mapAttrs (k: _: {
          type = "app";
          program = "${self.packages.${system}.default}/bin/${instantiate-default k}";
        }) apps;
        packages = {
          default = pkgs.stdenv.mkDerivation {
            inherit pname src version;
            buildPhase =
              let
                chmod = "${pkgs.coreutils}/bin/chmod";
                echo = "${pkgs.coreutils}/bin/echo";
                mkdir = "${pkgs.coreutils}/bin/mkdir";
                mv = "${pkgs.coreutils}/bin/mv";
                shebang = ''
                  #!${pkgs.bash}/bin/bash
                  set -eu
                  export JAX_ENABLE_X64=1
                '';
              in
              ''
                ${mkdir} -p $out/bin
                ${mv} ./* $out/
                ${builtins.foldl' (a: b: a + b) "" (
                  builtins.attrValues (
                    builtins.mapAttrs (
                      k: v:
                      let
                        bin = "$out/bin/${instantiate-default k}";
                      in
                      ''

                        ${echo} '${shebang}' > ${bin}
                        ${echo} "${v}" >> ${bin}
                        ${chmod} +x ${bin}
                      ''
                    ) apps
                  )
                )}
              '';
          };
        };
        devShells.default = pkgs.mkShell {
          JAX_ENABLE_X64 = "1";
          packages = lookup-pkg-sets [
            default-pkgs
            check-pkgs
            ci-pkgs
            dev-pkgs
          ] pkgs pypkgs;
        };
      }
    );
}
