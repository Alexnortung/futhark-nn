{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable"; # TODO: use unstable when futhark is updated
        flake-utils.url = "github:numtide/flake-utils";
        flake-utils.inputs.nixpkgs.follows = "nixpkgs";
    };
    outputs = { nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachDefaultSystem (system: {
        devShell = import ./shell.nix {
            inherit system;
            pkgs = nixpkgs.legacyPackages.${system};
        };
        packages = flake-utils.lib.flattenTree rec {
            mnist-data = import ./programs/fetch-mnist.nix {
                inherit system;
                pkgs = nixpkgs.legacyPackages.${system};
                lib = nixpkgs.lib;
            };
            mnist-images-futhark = import ./programs/mnist-futhark.nix {
                inherit system mnist-data;
                pkgs = nixpkgs.legacyPackages.${system};
                lib = nixpkgs.lib;
            };
            mnist-training-images-futhark = mnist-images-futhark.mnist-training-images;
            mnist-test-images-futhark = mnist-images-futhark.mnist-test-images;
        };
    });
}
