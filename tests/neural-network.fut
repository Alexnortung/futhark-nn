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


-- entry nn_compositon (input) =
--   let seed = 1
--   in nn.init_2d 3 3 seed
--   |> nn.conv_2d 2 2 2 2
--   |> nn.maxpool_2d 1 1
--   |> nn.forward input

-- ==
-- entry: nn_layer_composition
-- input {
--    [[[2.0, 3.0,7.0],[1.0, 1.0, 2.0],[5.0,4.0,3.0]]]
-- }
-- output {
--    [[-3.0430608262944023f64, -3.0430608262944023f64]]
-- }
entry nn_layer_composition (input) =
  let seed = 1
  let l1 = nn.conv.init_2d 2 2 2 2 seed
  let l2 = nn.mpool.init_2d 1 1
  let l2c = nn.dimension.from_2d_1d 1
  let l3 = nn.lin.init 1 3 (identity) seed
  let l4 = nn.conv.init_1d 2 2 seed
  in nn.init_2d 3 3 seed
  |> nn.add_layer l1
  |> nn.add_layer l2
  |> nn.add_layer l2c
  |> nn.add_layer l3
  |> nn.add_layer l4
  |> nn.forward input

-- ==
-- entry: nn_dimension_test
-- input {
--    [[1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64]]
-- }
-- output {
--    [[1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64]]
-- }
entry nn_dimension_test (input) =
  let seed = 1
  in nn.init_1d 8 seed
  |> nn.add_layer (nn.dimension.from_1d_3d 2 2 2)
  |> nn.add_layer (nn.dimension.from_3d_1d 8)
  |> nn.add_layer (nn.dimension.from_1d_2d 4 2)
  |> nn.add_layer (nn.dimension.from_2d_1d 8)
  |> nn.forward input
