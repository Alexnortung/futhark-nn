import "../layers/convolutional"
import "../layers/pooling"
import "../layers/fully-connected"
import "../util/activation-func"
import "../util/change-dimensions"

module myconv = convolutional f64
module mypool = pooling f64
module fcs = fully_connected_simple f64

-- ==
-- entry: composition_test
-- input {
--   [[[ 1.0, 2.0, 3.0,  1.0, 2.0],
--     [10.0, 9.0, 8.0, 10.0, 9.0],
--     [ 4.0, 5.0, 6.0, 4.0, 5.0],
--     [ 1.0, 2.0, 3.0,  1.0, 2.0],
--     [ 4.0, 5.0, 6.0, 4.0, 5.0]],
--    [[ 1.0, 2.0, 3.0, 1.0, 2.0],
--     [10.0, 9.0, 8.0, 10.0, 9.0],
--     [ 4.0, 5.0, 7.0, 4.0, 5.0],
--     [10.0, 9.0, 8.0, 10.0, 9.0],
--     [ 4.0, 5.0, 7.0, 4.0, 5.0]]]
--
--   [[1.0, 3.0],
--    [2.0, 9.0]]
--
--    3.0
--
--    2i64
--
--    2i64
--
--   [[1.0, 2.0, 3.0, 5.0],
--    [9.0, 7.0, 2.0, 4.0],
--    [1.0, 9.0, 8.0, 2.0]]
--
--    [3.0, 1.0, 3.0]
-- }
-- output {
--    [[ 893.0, 2213.0, 1899.0],
--     [1353.0, 2563.0, 2389.0]]
-- }

entry composition_test [k] [m] [n] [w1x] [w1y] [m3] [n3] (input: [k][m][n]f64) (w1: [w1x][w1y]f64) (b1: f64) (outx2: i64) (outy2: i64) (w3: [n3][m3]f64) (b3: [n3]f64) : [k][n3]f64 =
  let out = myconv.forward input b1 w1
  let out = mypool.forward out outx2 outy2
  let out = change_dimensions.from_2d_to_1d out
  let out = fcs.forward out (identity) w3 b3
  in out

