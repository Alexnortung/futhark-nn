import "../neural-network/neural-network-new"
import "../util/activation-func"
import "../layers/convolutional"

module nn = neural_network f64

-- ==
-- entry: nn_test
-- input {
--  [[[1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0]],
--   [[1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0]]]
--
--    1i32
-- }
-- output {
--    [[[18.974539242722912f64, 27.177478337886157f64, 35.38041743304941f64],
--      [18.974539242722912f64, 27.177478337886157f64, 35.38041743304941f64]],
--     [[18.974539242722912f64, 27.177478337886157f64, 35.38041743304941f64],
--      [18.974539242722912f64, 27.177478337886157f64, 35.38041743304941f64]]]
-- }

entry nn_test [k] (input: [k][6][5]f64) (seed: i32) =
  let n = nn.init_2d 6 5 seed
  let shape = nn.conv_2d_shape 3 2 n
  let n = nn.conv_2d shape.0 shape.1 3 2 n
  let shape = nn.conv_2d_shape 3 2 n
  let n = nn.conv_2d shape.0 shape.1 3 2 n
  in nn.forward input n

-- ==
-- entry: nn_linear_test
-- input {
--    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
-- }
-- output {
--    [[-18.80553453580382f64]]
-- }
entry nn_linear_test [k] (input: [k][]f64) =
  nn.init_1d 7 1
  |> nn.linear 5 (identity)
  |> nn.linear 6 (identity)
  |> nn.linear 1 (identity)
  |> nn.forward input

-- entry: nn_maxpool_2d_test
-- input {
--  [
--   [[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]],
--  ]
-- }
-- output { [
--   [[2.0, 4.0, 2.0, 4.0],
--    [2.0, 4.0, 2.0, 4.0]]
-- ] }
entry nn_maxpool_2d_test (input: [][][]f64) =
  let n = nn.init_2d 8 8 1
  let shape = nn.maxpool_2d_shape 2 2 n
  let n = nn.maxpool_2d shape.0 shape.1 n
  let shape = nn.maxpool_2d_shape 2 1 n
  in nn.maxpool_2d shape.0 shape.1 n
    |> nn.forward input

-- ==
-- entry: nn_add_layer_test
-- input {[
--        [[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 6.0]],
--        [[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 7.0]]]
--
--        3.0
--      
--        [[1.0, 3.0],
--         [2.0, 9.0]]
-- }
-- output {[
--         [[111.0, 104.0],
--          [ 93.0, 100.0]],
--         [[111.0, 104.0],
--          [ 93.0, 109.0]]
--         ]
-- }

entry nn_add_layer_test (input: [][][]f64 ) (b1: f64) (w1: [][]f64) =
  let seed = 1
  in nn.init_2d 3 3 seed
  |> nn.add_layer (nn.conv.init_2d 2 2 2 2 seed
                   |> nn.conv.set_weights w1
                   |> nn.conv.set_bias b1)
  |> nn.forward input
