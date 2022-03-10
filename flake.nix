{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
        futhark.url = "github:Alexnortung/futhark/clean-ad?dir=nix";
    };
    outputs = { futhark, nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachDefaultSystem (system: {
        devShell = import ./shell.nix {
            inherit system;
            pkgs = nixpkgs.legacyPackages.${system};
            futhark = futhark;
        };
    });
}
