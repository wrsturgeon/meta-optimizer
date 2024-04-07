{
  description = "Quantifying performance of machine-learning optimizers like RMSProp & Adam.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      flake-utils,
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
          p: with p; [
            (jax.overridePythonAttrs (
              old:
              old
              // {
                doCheck = false;
                propagatedBuildInputs = old.propagatedBuildInputs ++ [ p.jaxlib-bin ];
              }
            ))
            beartype
            jaxtyping
            matplotlib
          ];
        check-pkgs =
          p: with p; [
            hypothesis
            pytest
          ];
        ci-pkgs =
          p: with p; [
            black
            coverage
            mypy
          ];
        dev-pkgs = p: with p; [ python-lsp-server ];
        lookup-pkg-sets = ps: p: builtins.concatMap (f: f p) ps;
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps)}/bin/python";
        instantiate-default = s: if s == "default" then pname else s;
        apps = {
          ci =
            let
              python = python-with [
                default-pkgs
                check-pkgs
                ci-pkgs
              ];
            in
            ''
              rm -fr result
              ${python} -m black --check .
              ${python} -m mypy .
              ${python} -m coverage run --omit='/nix/*' -m pytest -Werror test.py
              ${python} -m coverage report -m --fail-under=100
            '';
          default = ''
            ${python-with [ default-pkgs ]} $out/main.py
          '';
          plot = ''
            ${python-with [ default-pkgs ]} $out/plot.py
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
            # checkPhase =
            #   let
            #     python = python-with [
            #       default-pkgs
            #       check-pkgs
            #     ];
            #   in
            #   ''
            #     cd $out
            #     ${python} -m pytest -Werror test.py
            #   '';
            # doCheck = true;
          };
        };
        devShells.default = pkgs.mkShell {
          packages = lookup-pkg-sets [
            default-pkgs
            check-pkgs
            ci-pkgs
            dev-pkgs
          ] pypkgs;
        };
      }
    );
}
