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
        buildInputs = with pypkgs; [ jax jaxlib-bin python ];
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          inherit buildInputs pname src version;
          buildPhase = ":";
          installPhase = ''
            mkdir -p $out/bin
            mv ./* $out/
            echo '#!/usr/bin/env bash' > $out/bin/${pname}
            echo "cd $out/src" >> $out/bin/${pname}
            echo "${pypkgs.python}/bin/python $out/src/main.py" >> $out/bin/${pname}
            chmod +x $out/bin/${pname}
          '';
        };
        devShells.default = pkgs.mkShell {
          packages = buildInputs ++ (with pypkgs; [ python-lsp-server ]);
        };
      });
}
