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
        pypkgs = pkgs.python312Packages;
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
        ci-script =
          let
            python = python-with [
              default-pkgs
              check-pkgs
              ci-pkgs
            ];
          in
          ''
            set -eux
            ${python} -m black --check .
            ${python} -m mypy .
            ${python} -m coverage run --omit "/nix/*" -m pytest -Werror test.py
            ${python} -m coverage report -m
            export COVPCT=$(${python} -m coverage report -m | tail -n 1 | tr -s " " | cut -d " " -f 4)
            if [ "''${COVPCT}" != "100%" ]; then
              echo "Coverage reported ''${COVPCT} overall, but we expected 100%"
              exit 1
            fi
          '';
        buildAndRun = double-quotes: exec: ''
          mkdir -p $out/bin
          mv ./* $out/
          echo '#!${pkgs.bash}/bin/bash' > $out/bin/${pname}
          echo ${if double-quotes then "\"${exec}\"" else "'${exec}'"} >> $out/bin/${pname}
          chmod +x $out/bin/${pname}
        '';
      in
      {
        packages = {
          apps = builtins.mapAttrs (k: _: {
            type = "app";
            program = "${self.packages.${system}.${k}}/bin/${pname}";
          }) self.packages.${system};
          default = pkgs.stdenv.mkDerivation {
            inherit pname src version;
            buildPhase = buildAndRun true ''
              cd $out
              ${python-with [ default-pkgs ]} $out/main.py
            '';
            checkPhase =
              let
                python = python-with [
                  default-pkgs
                  check-pkgs
                ];
              in
              ''
                cd $out
                ${python} -m pytest -Werror test.py
              '';
            doCheck = true;
          };
          ci = pkgs.stdenv.mkDerivation {
            inherit pname src version;
            buildPhase = buildAndRun false ci-script;
            checkPhase = ":";
            doCheck = false;
          };
        };
        devShells.default = pkgs.mkShell {
          packages = lookup-pkg-sets (default-pkgs ++ check-pkgs ++ ci-pkgs ++ dev-pkgs) pypkgs;
        };
      }
    );
}
