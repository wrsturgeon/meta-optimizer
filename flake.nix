{
  description =
    "Quantifying performance of machine-learning optimizers like RMSProp & Adam.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs = { flake-utils, nixpkgs, self }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pypkgs = pkgs.python311Packages;
        pname = "meta-optimizer";
        version = "0.0.1";
        src = ./.;
        # TODO: Use pylyzer when 1.76.0+ supported
        jax = pypkgs.jax.overridePythonAttrs (old: {
          doCheck = false;
          propagatedBuildInputs = old.propagatedBuildInputs
            ++ [ pypkgs.jaxlib-bin ];
        });
        propagatedBuildInputs = [ jax ] ++ (with pypkgs; [ python ]);
        checkInputs = with pypkgs; [ hypothesis pytest ];
        shellInputs = with pypkgs; [ black python-lsp-server ];
        buildAndRun = exec: ''
          mkdir -p $out/bin
          mv ./* $out/
          echo '#!/usr/bin/env bash' > $out/bin/${pname}
          echo "cd $out/src" >> $out/bin/${pname}
          echo "${exec}" >> $out/bin/${pname}
          chmod +x $out/bin/${pname}
        '';
        checkPhase = ''
          ${pypkgs.pytest}/bin/pytest $out/src/test.py
        '';
        derivationSettings = {
          inherit propagatedBuildInputs pname src version;
          buildPhase =
            buildAndRun "${pypkgs.python}/bin/python $out/src/main.py";
        };
      in {
        packages = {
          default = pkgs.stdenv.mkDerivation (derivationSettings // {
            inherit checkInputs checkPhase;
            doCheck = true;
          });
          ci = pkgs.stdenv.mkDerivation (derivationSettings // {
            propagatedBuildInputs = propagatedBuildInputs ++ checkInputs;
            buildPhase = buildAndRun ''
              export GITHUB_CI=1
              ${checkPhase}
            '';
          });
        };
        devShells.default = pkgs.mkShell {
          packages = propagatedBuildInputs ++ checkInputs ++ shellInputs;
        };
      });
}
