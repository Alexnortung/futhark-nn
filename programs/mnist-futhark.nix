{ pkgs, lib, mnist-data, ... }:

let
    numberSuffix = ".0";
    # dataType = "f64";
    dataType = "";
    buildTraining = true;
    buildTest = true;
    numTrainingImages = 20;
    numTestImages = 20;
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
        head -c $((16 + ${toString numTrainingImages} * $ROWS * $COLS)) $UNZIPPED \
            | od -An -v --width=1 -t u1 --endian=big --skip-bytes=16 \
            | ${pkgs.jq}/bin/jq -scM '. as $input | [range(0; length; '$ROWS')] | map($input[. : . + '$ROWS']) | . as $input | [range(0; length; '$COLS')] | map($input[. : . + '$COLS']) | map([.])' \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > training-images-futhark.json

        # training labels
        UNZIPPED="train-labels-idx1-ubyte"
        echo "Building training labels"
        head -c $((8 + ${toString numTrainingImages})) $UNZIPPED \
            | od -An -v --width=1 -t u1 --endian=big --skip-bytes=8 \
            | ${pkgs.jq}/bin/jq -scM \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > training-labels-futhark.json

        # test images
        UNZIPPED="t10k-images-idx3-ubyte"
        echo "Building test images"
        head -c $((16 + ${toString numTestImages} * $ROWS * $COLS)) $UNZIPPED \
            | od -An -v --width=1 -t u1 --endian=big --skip-bytes=16 \
            | ${pkgs.jq}/bin/jq -scM '. as $input | [range(0; length; '$ROWS')] | map($input[. : . + '$ROWS']) | . as $input | [range(0; length; '$COLS')] | map($input[. : . + '$COLS']) | map([.])' \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > test-images-futhark.json

        # test labels
        UNZIPPED="t10k-labels-idx1-ubyte"
        echo "Building test labels"
        head -c $((8 + ${toString numTestImages})) $UNZIPPED \
            | od -An -v --width=1 -t u1 --endian=big --skip-bytes=8 \
            | ${pkgs.jq}/bin/jq -scM \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1${numberSuffix}${dataType}/g' \
            > test-labels-futhark.json

        # cat training-images-futhark training-labels-futhark > training-data
        # cat test-images-futhark test-labels-futhark > test-data
    '';

    installPhase = ''
        mkdir -p $out
        cp training-images-futhark.json $out/
        cp training-labels-futhark.json $out/
        cp test-images-futhark.json $out/
        cp test-labels-futhark.json $out/
    '';
}
