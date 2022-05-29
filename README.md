# Futhark neural network

Work in progress, some things may not work yet.

This is a futhark package that implements neural networks in Futhark.

## Development

For constistency in development this repository uses nix, which is the only requirement for development.
Please use nix version `>= 2.7`

To get a shell with just the version of Futhark use

```bash
nix develop
```

If you need a shell with cudatoolkit, you can use

```bash
export NIXPKGS_ALLOW_UNFREE=1
nix develop .\#cuda --impure
```

The impure flag needs be used if you do not allow unfree packages by default.
