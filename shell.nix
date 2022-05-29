{ pkgs ? <nixpkgs>, system, with-cuda ? false, ... }:

pkgs.stdenvNoCC.mkDerivation {
    name = (pkgs.lib.optionalString with-cuda "cuda-") + "env-shell";
    buildInputs = [
        pkgs.futhark
    ] ++ (pkgs.lib.optionals with-cuda [
        pkgs.cudatoolkit
        pkgs.linuxPackages.nvidia_x11
    ]);

    shellHook = pkgs.lib.optionalString with-cuda ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
    '';
}
