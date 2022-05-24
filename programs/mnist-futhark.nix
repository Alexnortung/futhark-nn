{ pkgs, lib, mnist-data, ... }:

let
    numberSuffix = ".0";
    dataType = "f64";
in
pkgs.stdenvNoCC.mkDerivation {
    name = "mnist-futhark";

    src = mnist-data;

    unpackPhase = ''
        cp $src/*.gz ./
        ${pkgs.gzip}/bin/gunzip *.gz
    '';

    buildPhase = ''
        # training images
        UNZIPPED="train-images-idx3-ubyte"
        od -An --read-bytes=16 -t u4 --output-duplicates --width=4 --endian=big $UNZIPPED | tr --delete ' ' > header-lines

        NUM_IMAGES=$(head --lines=2 header-lines | tail --lines=1) # Not needed right now
        ROWS=$(head --lines=3 header-lines | tail --lines=1)
        COLS=$(tail --lines=1 header-lines)

        echo "Building training images"
        od -An -v --width=1 -t u1 --endian=big --skip-bytes=16 $UNZIPPED \
            | ${pkgs.jq}/bin/jq -cM '[inputs] | . as $input | [range(0; length; '$ROWS')] | map($input[. : . + '$ROWS']) | . as $input | [range(0; length; '$COLS')] | map($input[. : . + '$COLS']) | map([.])' \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > training-images-futhark

        # test images
        UNZIPPED="t10k-images-idx3-ubyte"
        echo "Building test images"
        od -An -v --width=1 -t u1 --endian=big --skip-bytes=16 $UNZIPPED \
            | ${pkgs.jq}/bin/jq -cM '[inputs] | . as $input | [range(0; length; '$ROWS')] | map($input[. : . + '$ROWS']) | . as $input | [range(0; length; '$COLS')] | map($input[. : . + '$COLS']) | map([.])' \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > test-images-futhark

        # training labels
        UNZIPPED="train-labels-idx1-ubyte"
        echo "Building training labels"
        od -An -v --width=1 -t u1 --endian=big --skip-bytes=8 $UNZIPPED \
            | ${pkgs.jq}/bin/jq -cM '[inputs] | map([.])' \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > training-labels-futhark

        # test labels
        UNZIPPED="t10k-labels-idx1-ubyte"
        echo "Building test labels"
        od -An -v --width=1 -t u1 --endian=big --skip-bytes=8 $UNZIPPED \
            | ${pkgs.jq}/bin/jq -cM '[inputs] | map([.])' \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > test-labels-futhark

        cat training-images-futhark training-labels-futhark > training-data
        cat test-images-futhark test-labels-futhark > test-data
    '';

    installPhase = ''
        mkdir -p $out
        cp training-data $out/
        cp test-data $out/
    '';
}
