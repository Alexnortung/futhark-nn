{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs"; # TODO: use unstable when futhark is updated
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = { nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachDefaultSystem (system: {
        devShell = import ./shell.nix {
            inherit system;
            pkgs = nixpkgs.legacyPackages.${system};
        };
    });
}
