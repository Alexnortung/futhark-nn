{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
        flake-utils.inputs.nixpkgs.follows = "nixpkgs";
    };
    outputs = { nixpkgs, flake-utils, ... }: 
    flake-utils.lib.eachDefaultSystem (system: {
        devShells = flake-utils.lib.flattenTree rec {
            default = import ./shell.nix {
                inherit system;
                pkgs = nixpkgs.legacyPackages.${system};
            };
            cuda = import ./shell.nix {
                inherit system;
                pkgs = nixpkgs.legacyPackages.${system};
                with-cuda = true;
            };
        };
        # devShell.cuda = import ./shell.nix {
        #     inherit system;
        #     pkgs = nixpkgs.legacyPackages.${system};
        #     with-cuda = true;
        # };

        packages = flake-utils.lib.flattenTree rec {
            mnist-data = import ./programs/fetch-mnist.nix {
                inherit system;
                pkgs = nixpkgs.legacyPackages.${system};
                lib = nixpkgs.lib;
            };
            mnist-futhark = import ./programs/mnist-futhark.nix {
                inherit system mnist-data;
                pkgs = nixpkgs.legacyPackages.${system};
                lib = nixpkgs.lib;
            };
        };
    });
}
