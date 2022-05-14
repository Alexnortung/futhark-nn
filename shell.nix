{ pkgs, system, ... }:

pkgs.mkShell {
    buildInputs = [
        pkgs.futhark
    ];
}
