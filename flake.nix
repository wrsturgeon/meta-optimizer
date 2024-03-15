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
        jax = pypkgs.jax.overridePythonAttrs (old: {
          doCheck = false;
          propagatedBuildInputs = old.propagatedBuildInputs
            ++ [ pypkgs.jaxlib-bin ];
        });
        buildInputs = [ jax ] ++ (with pypkgs; [ hypothesis python ]);
        checkInputs = with pypkgs; [ pytest ];
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
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          inherit buildInputs checkInputs checkPhase pname src version;
          buildPhase =
            buildAndRun "${pypkgs.python}/bin/python $out/src/main.py";
          doCheck = true;
        };
        devShells.default = pkgs.mkShell {
          packages = buildInputs ++ checkInputs ++ shellInputs;
        };
      });
}
