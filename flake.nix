{
  description =
    "Quantifying performance of machine-learning optimizers like SGD, RMSprop & Adam.";
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
        buildInputs = with pypkgs; [ hypothesis jax jaxlib-bin python ];
        testInputs = with pypkgs; [ pytest ];
        shellInputs = with pypkgs; [ black python-lsp-server ];
        buildPhase = ":";
        installAndRun = exec: ''
          mkdir -p $out/bin
          mv ./* $out/
          echo '#!/usr/bin/env bash' > $out/bin/${pname}
          echo "cd $out/src" >> $out/bin/${pname}
          echo "${exec}" >> $out/bin/${pname}
          chmod +x $out/bin/${pname}
        '';
      in {
        packages = {
          default = pkgs.stdenv.mkDerivation {
            inherit buildInputs buildPhase pname src version;
            installPhase =
              installAndRun "${pypkgs.python}/bin/python $out/src/main.py";
          };
          test = pkgs.stdenv.mkDerivation {
            inherit buildPhase pname src version;
            buildInputs = buildInputs ++ testInputs;
            installPhase =
              installAndRun "${pypkgs.pytest}/bin/pytest $out/src/test.py";
          };
        };
        devShells.default =
          pkgs.mkShell { packages = buildInputs ++ testInputs ++ shellInputs; };
      });
}
