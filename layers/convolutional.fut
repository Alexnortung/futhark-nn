import "../util/linalg"
import "../util/weight-initialization"
import "types"

module convolutional (R:real) = {
  type t = R.t

  type options 'shape = {
    stride: shape,
    padding: shape
  }
  type options_1d = options (shape_1d)
  type options_2d = options (shape_2d)
  type batch_1d [k] [channels] [n] = [k][channels][n]t
  type batch_2d [k] [channels] [m] [n] = [k][channels][m][n]t
  type kernel_1d [channels] [a] = [channels][a]t
  type kernel_2d [channels] [a] [b] = [channels][a][b]t
  type weight_bias_1d [channels] [a] = (kernel_1d [channels] [a], [channels]t)
  type weight_bias_2d [channels] [a] [b] = (kernel_2d [channels] [a] [b], [channels]t)
  type^ layer_type_1d [k] [in_channels] [out_channels] [n] [a] [out_n] = layer_type t options_1d (batch_1d [k] [in_channels] [n]) (weight_bias_1d [out_channels] [a]) shape_2d (batch_1d [k] [out_channels] [out_n])
  type^ layer_type_2d [k] [in_channels] [out_channels] [m] [n] [a] [b] [out_m] [out_n] = layer_type t options_2d (batch_2d [k] [in_channels] [m] [n]) (weight_bias_2d [out_channels] [a] [b]) shape_3d (batch_2d [k] [out_channels] [out_m] [out_n])

  module lalg = mk_linalg R

  module wi = weight_init R

  let default_options_1d : options_1d = {
    stride = 1,
    padding = 0
  }
  let default_options_2d : options_2d = {
    stride = (1, 1),
    padding = (0, 0)
  }

  def forward_1d [k] [in_channels] [out_channels] [n] [a] -- k batches, m times n input nodes and a times b filter size
    (output_n: i64)
    (input: batch_1d [k] [in_channels] [n])
    (activation_func: activation_type t)
    (bias: [out_channels]t)
    (filter_weights: kernel_1d [out_channels] [a]) : batch_1d [k] [out_channels] [output_n] =
      let xs = iota output_n
      in
      map (\batch ->
        map2 (\filter_channel bias_channel ->
          map (\x -> 
            map (\channel ->
              let input_slice = channel[x:x+a] :> [a]t
              let dotted : t = lalg.dotprod input_slice filter_channel
              in R.(dotted)
            ) batch
            |> R.(sum)
            |> R.((+) bias_channel)
            |> activation_func
          ) xs
        ) filter_weights bias
      ) input

  def forward_2d [k] [in_channels] [out_channels] [m] [n] [a] [b] -- k batches, m times n input nodes and a times b filter size
    (output_m: i64)
    (output_n: i64)
    (input: batch_2d [k] [in_channels] [m] [n])
    (activation_func: activation_type t)
    (bias: [out_channels]t)
    (filter_weights: kernel_2d [out_channels] [a] [b]) : batch_2d [k] [out_channels] [output_m] [output_n] =
      let xs = iota output_m -- m - a + 1
      let ys = iota output_n -- n - b + 1
      let c = a * b
      in
      map (\batch ->
        map2 (\filter_channel bias_channel ->
          let flat_weights = flatten_to c filter_channel
          in
          map (\x -> 
            map (\y ->
              map (\channel ->
                let input_slice = channel[x:x+a,y:y+b]
                let flat_slice = flatten_to c input_slice
                let dotted : t = lalg.dotprod flat_slice flat_weights
                in R.(dotted)
              ) batch
              |> R.(sum)
              |> R.((+) bias_channel)
              |> activation_func
            ) ys
          ) xs
        ) filter_weights bias
      ) input

  def apply_optimize_1d [out_channels] [n]
    (_options: options_1d)
    (apply_func_record: optimizer_apply_record t)
    ((current_weights, current_bias): weight_bias_1d [out_channels] [n])
    ((gradient_weights, gradient_bias): weight_bias_1d [out_channels] [n])
    : weight_bias_1d [out_channels] [n] =
      let new_weights = apply_func_record.apply_2d out_channels n current_weights gradient_weights
      let new_bias = apply_func_record.apply_1d out_channels current_bias gradient_bias
      in (new_weights, new_bias)

  def apply_optimize_2d [out_channels] [m] [n]
    (_options: options_2d)
    (apply_func_record: optimizer_apply_record t)
    ((current_weights, current_bias): weight_bias_2d [out_channels] [m] [n])
    ((gradient_weights, gradient_bias): weight_bias_2d [out_channels] [m] [n])
    : weight_bias_2d [out_channels] [m] [n] =
      let new_weights = apply_func_record.apply_3d out_channels m n current_weights gradient_weights
      let new_bias = apply_func_record.apply_1d out_channels current_bias gradient_bias
      in (new_weights, new_bias)

  def generate_weights_1d (seed: i32) (out_channels: i64) (a: i64) : weight_bias_1d [out_channels] [a] =
    let weights = wi.gen_2d out_channels a seed
    let bias = wi.gen_1d out_channels seed
    in (weights, bias)

  def generate_weights_2d (seed: i32) (out_channels: i64) (a: i64) (b: i64) : weight_bias_2d [out_channels] [a] [b] =
    let weights = wi.gen_3d out_channels a b seed
    let bias = wi.gen_1d out_channels seed
    in (weights, bias)

  def layer_new_forward_1d [k] [in_channels] [out_channels] [n] [a]
    (output_n: i64)
    (activation_func: activation_type t)
    (_options: options_1d)
    (input: batch_1d [k] [in_channels] [n])
    (wb: weight_bias_1d [out_channels] [a]) : batch_1d [k] [out_channels] [output_n] =
      let (kernel, bias) = wb
      in forward_1d output_n input activation_func bias kernel

  def layer_new_forward_2d [k] [in_channels] [out_channels] [m] [n] [a] [b]
    (output_m: i64)
    (output_n: i64)
    (activation_func: activation_type t)
    (_options: options_2d)
    (input: batch_2d [k] [in_channels] [m] [n])
    (wb: weight_bias_2d [out_channels] [a] [b]) : batch_2d [k] [out_channels] [output_m] [output_n] =
      let (kernel, bias) = wb
      in forward_2d output_m output_n input activation_func bias kernel

  def init_1d [k] [n]
    (output_n: i64)
    (in_channels: i64)
    (out_channels: i64)
    (kernel_sz: i64)
    (activation_func: activation_type t)
    (seed: i32)
    : layer_type_1d [k] [in_channels] [out_channels] [n] [kernel_sz] [output_n] =
      let forward = layer_new_forward_1d output_n activation_func
      let options = default_options_1d
      let weights = generate_weights_1d seed out_channels kernel_sz
      in { forward, apply_optimize = apply_optimize_1d,
           options, weights,
           shape = (out_channels, output_n) }

  def init_2d [k] [m] [n]
    (output_m: i64)
    (output_n: i64)
    (in_channels: i64)
    (out_channels: i64)
    (kernel_x: i64)
    (kernel_y: i64)
    (activation_func: activation_type t)
    (seed: i32)
    : layer_type_2d [k] [in_channels] [out_channels] [m] [n] [kernel_x] [kernel_y] [output_m] [output_n] =
      let forward = layer_new_forward_2d output_m output_n activation_func
      let options = default_options_2d
      let weights = generate_weights_2d seed out_channels kernel_x kernel_y
      in { forward, apply_optimize = apply_optimize_2d,
           options, weights,
           shape = (out_channels, output_m, output_n) }

  -- TODO: set reutrn type such that bias must have the same type as the current bias
  def set_bias -- [k] [m] [n] [a] [b] [out_m] [out_n]
    (bias)
    (layer)
    =
      let { forward, apply_optimize, options, shape, weights = (weights, _)} = layer
      in  { forward, apply_optimize, options, shape, weights = (weights, bias)}

  -- TODO: set reutrn type such that bias must have the same type as the current bias
  def set_weights -- [k] [m] [n] [a] [b]
    (weights)
    (layer)
    =
      let { forward, apply_optimize, options, shape, weights = (_, bias)} = layer
      in  { forward, apply_optimize, options, shape, weights = (weights, bias)}

  def forward_layer (input) (layer) =
    let { forward, apply_optimize = _, options, weights, shape = _ } = layer
    in forward options input weights

  -- TODO: add functions for setting padding and stride
}

-- TESTS

local module test_conv = convolutional f64

-- ==
-- entry: conv conv_layer
-- input {[
--        [[[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 6.0]]],
--        [[[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 7.0]]]]
--
--        [3.0]
--      
--        [[[1.0, 3.0],
--         [2.0, 9.0]]]
-- }
-- output {[
--         [[[111.0, 104.0],
--          [ 93.0, 100.0]]],
--         [[[111.0, 104.0],
--          [ 93.0, 109.0] ]]
--         ]
-- }

entry conv [k] (input: [k][][3][3]f64) (bias: []f64) (weights: [][2][2]f64) : [k][][2][2]f64 =
  test_conv.forward_2d 2 2 input (id) bias weights

entry conv_layer [k] [a] [b] (input: [k][][][]f64) (bias: []f64) (weights: [][a][b]f64) : [k][][][]f64 =
  test_conv.init_2d 2 2 1 1 a b (id) 1
  |> test_conv.set_bias bias
  |> test_conv.set_weights weights
  |> test_conv.forward_layer input

------- ==
-- entry: conv_layer_channels
-- input {
--    [
--    [
--      [[1.0, 2.0], [3.0, 4.0]],
--      [[5.0, 6.0], [7.0, 8.0]]
--    ]
--    ]
--
--    [2.0, 3.0]
--
--    [
--    [
--      [10.0, 2.0],
--      [-1.0, -2.0]
--    ],
--    [
--      [20.0, 4.0],
--      [-3.0, -4.0]
--    ]
--    ]
--
-- }
-- output {
--    [[[[44.000000f64]], [[77.000000f64]]]]
-- }
entry conv_layer_channels [k] (input: [k][2][2][2]f64) (bias: [2]f64) (weights: [2][2][2]f64) : [k][2][1][1]f64 =
  test_conv.init_2d 1 1 2 2 2 2 (id) 1
  |> test_conv.set_bias bias
  |> test_conv.set_weights weights
  |> test_conv.forward_layer input

entry conv_1d (input: [][][]f64) (bias: []f64) (weights: [][]f64) : [][][]f64 =
  test_conv.init_1d 2 2 2 1 (id) 1
  |> test_conv.set_bias bias
  |> test_conv.set_weights weights
  |> test_conv.forward_layer input
