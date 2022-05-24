{ pkgs, lib, mnist-data, ... }:

let
    mnist-images = pkgs.writeShellScript "mnist-images-futhark" ''
        set -e
        UNZIPPED=/tmp/mnist-unzipped-$$$$.bin
        HEADER=/tmp/mnist-header-$$$$.bin
        HEADER_LINES=/tmp/mnist-header-$$$$.txt
        IMAGES_BIN=/tmp/mnist-images-$$$$.bin

        ${pkgs.gzip}/bin/gunzip --stdout $1 > $UNZIPPED
        od -An --read-bytes=16 -t u4 --output-duplicates --width=4 --endian=big < $UNZIPPED | tr --delete ' ' > $HEADER_LINES

        NUM_IMAGES=$(head --lines=2 $HEADER_LINES | tail --lines=1) # Not needed right now
        ROWS=$(head --lines=3 $HEADER_LINES | tail --lines=1)
        COLS=$(head --lines=4 $HEADER_LINES | tail --lines=1)

        tail -c +16 $UNZIPPED \
            | od -An -v --width=1 -t u1 --endian=big --skip-bytes=16 \
            | ${pkgs.jq}/bin/jq -cM '[inputs] | . as $input | [range(0; length; '$ROWS')] | map($input[. : . + '$ROWS']) | . as $input | [range(0; length; '$COLS')] | map($input[. : . + '$COLS']) | map([.])' \
            | ${pkgs.gnused}/bin/sed -E 's/([0-9]+)/\1.0f16/g'
    '';
    mnist-labels = pkgs.writeShellScript "mnist-labels-futhark" ''

    '';
in
{
    mnist-training-images = pkgs.writeShellScriptBin "mnist-training-images-futhark" ''
        ${mnist-images} ${mnist-data}/train-images-idx3-ubyte.gz
    '';
    mnist-test-images = pkgs.writeShellScriptBin "mnist-test-images-futhark" ''
        ${mnist-images} ${mnist-data}/t10k-images-idx3-ubyte.gz
    '';
}
