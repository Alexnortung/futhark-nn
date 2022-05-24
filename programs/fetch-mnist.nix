{ pkgs, lib, ... }:

let 
    srcs = {
        train-images = pkgs.fetchurl {
            url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
            sha256 = "sha256-RA/Kv3PMVG+iFHXoHqNwJlYF9WviEKQCTSyo8gNSNgk=";
        };
        train-labels = pkgs.fetchurl {
            url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
            sha256 = "sha256-NVJTSgpVi77WrtMrMMSVzKI9Vn7FLKyL4aBzDoAQJVw=";
        };
        test-images = pkgs.fetchurl {
            url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
            sha256 = "sha256-jUIsewocHHkkWlvPB/6G4z7q/ueSuEWErsJ29aLbxOY=";
        };
        test-labels = pkgs.fetchurl {
            url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
            sha256 = "sha256-965g+S4A7G3r0jpgiMMdvSNx7KP/oN7677JZkkIErsY=";
        };
    };
in
pkgs.stdenvNoCC.mkDerivation {
    name = "mnist-data";

    srcs = [
        srcs.train-images
        srcs.train-labels
        srcs.test-images
        srcs.test-labels
    ];

    # dontUnpack = true;
    unpackPhase = ''
        runHook preUnpack

        for _src in $srcs; do
            cp "$_src" $(stripHash "$_src")
        done

        runHook postUnpack
    '';

    installPhase = ''
        runHook preInstall

        mkdir -p $out
        cp ./*.gz $out

        runHook postInstall
    '';
}
