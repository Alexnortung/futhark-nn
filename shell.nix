{ pkgs, futhark, ... }:

pkgs.mkShell {
    buildInputs = [
        futhark
    ];
}
