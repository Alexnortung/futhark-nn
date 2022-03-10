{ pkgs, futhark, system, ... }:

pkgs.mkShell {
    buildInputs = [
        futhark.packages.${system}.futhark
    ];
}
