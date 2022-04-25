import "../layers/convolutional"
import "../layers/linear"
import "../layers/pooling"
import "../layers/dimension"
import "../layers/types"

module neural_network (R:real) = {
  type t = R.t

  module conv = convolutional R
  module lin = linear R
  module mpool = pooling R
  module dimension = dimension

  def compose_forward 'input_type 'prev_wbs 'layer_input 'current_wbs 'output
    (prev_forward: input_type -> prev_wbs -> layer_input)
    (layer_forward: layer_input -> current_wbs -> output)
    (input: input_type)
    (wbs: (current_wbs, prev_wbs)) =
      let (current_wb, prev_wbs) = wbs
      let prev_result = prev_forward input prev_wbs
      in layer_forward prev_result current_wb

  def init_forward (x) _ = x

  def init_1d [k] (n: i64) (seed: i32) : nn_type shape_1d ([k][n]t) ([k][n]t) () () =
    {
      seed,
      shape = n,
      forward = init_forward,
      weights = ((), ())
    }

  def init_2d [k] (m: i64) (n: i64) (seed: i32) : nn_type shape_2d ([k][m][n]t) ([k][m][n]t) () () =
    {
      seed,
      shape = (m, n),
      forward = init_forward,
      weights = ((), ())
    }

  def add_layer 'nn_input 'l_input 'l_options  'l_wb 'l_shape 'l_out 'nn_shape 'prev_current_weight 'prev_rest_weight
    (layer: layer_type l_options l_input l_wb l_shape l_out)
    (network: nn_type nn_shape nn_input l_input prev_current_weight prev_rest_weight)
    : nn_type l_shape nn_input l_out l_wb (prev_current_weight, prev_rest_weight) =
    let { forward = layer_forward, options = layer_options, weights = layer_weights, shape = layer_shape } = layer
    let { shape = _, weights, forward, seed } = network
    let new_forward = compose_forward forward (layer_forward layer_options)
    in {
      seed,
      shape = layer_shape,
      weights = (layer_weights, weights),
      forward = new_forward
    }

  def conv_2d_shape 'input 'output 'cw 'rw (kernel_m: i64) (kernel_n: i64) (network: nn_type shape_2d input output cw rw) : shape_2d =
    let shape = network.shape
    let output_m = shape.0 - kernel_m + 1
    let output_n = shape.1 - kernel_n + 1
    in (output_m, output_n)

  def conv_2d 'input_type  'prev_current_weight 'prev_rest_weight 'current_weight
    [k] [prev_m] [prev_n]
    (output_m: i64)
    (output_n: i64)
    (kernel_m: i64)
    (kernel_n: i64)
    (network: nn_type shape_2d input_type ([k][prev_m][prev_n]t) prev_current_weight prev_rest_weight)
    : nn_type shape_2d input_type ([k][output_m][output_n]t) ([kernel_m][kernel_n]t, t) (prev_current_weight, prev_rest_weight)
    =
      let { shape = _, weights, forward, seed } = network
      let layer = conv.init_2d output_m output_n kernel_m kernel_n seed
      let { forward = layer_forward, options = layer_options, weights = layer_weights, shape = _} = layer
      let new_forward = compose_forward forward (layer_forward layer_options)
      in {
        seed,
        shape = (output_m, output_n),
        weights = (layer_weights, weights),
        forward = new_forward
      }

  def linear 'rest_weights 'input_type 'prev_current_weight 'prev_rest_weight
    [k] [m]
    -- (m: i64)
    (n: i64)
    (activation: t -> t)
    (network: nn_type shape_1d input_type (lin.input_type [k] [m]) prev_current_weight prev_rest_weight)
    : nn_type shape_1d input_type (lin.output_type [k] [n]) (lin.weights_and_bias [m] [n]) (prev_current_weight, prev_rest_weight) =
      let { shape = _, weights, forward, seed } = network
      let layer = lin.init m n activation seed
      let { forward = layer_forward, options = layer_options, weights = layer_weights, shape = _} = layer
      let new_forward = compose_forward forward (layer_forward layer_options)
      in {
        seed,
        shape = (n),
        weights = (layer_weights, weights),
        forward = new_forward
      }

  def maxpool_2d_shape 'input 'output 'cw 'rw (window_m: i64) (window_n: i64) (network: nn_type shape_2d input output cw rw) : shape_2d =
    let (m, n) = network.shape
    in (m / window_m, n / window_n)

  def maxpool_2d 'input_type  'prev_current_weight 'prev_rest_weight 'current_weight
    [k] [prev_m] [prev_n]
    (output_m: i64)
    (output_n: i64)
    (network: nn_type shape_2d input_type ([k][prev_m][prev_n]t) prev_current_weight prev_rest_weight)
    : nn_type shape_2d input_type ([k][output_m][output_n]t) () (prev_current_weight, prev_rest_weight)
    =
      let { shape = _, weights, forward, seed } = network
      let layer = mpool.init_2d output_m output_n
      let { forward = layer_forward, options = layer_options, weights = layer_weights, shape = _ } = layer
      let new_forward = compose_forward forward (layer_forward layer_options)
      in {
        seed,
        shape = (output_m, output_n),
        weights = (layer_weights, weights),
        forward = new_forward
      }

  -- def from_1d_2d (output_m) (output_n) (network) =
  --   let layer = dimension.from_1d_2d output_m output_n
  --   in add_layer layer network
  --
  -- def from_1d_3d (output_l) (output_m) (output_n) (network) =
  --   let layer = dimension.from_1d_3d output_l output_m output_n
  --   in add_layer layer network
  --
  -- def from_2d_1d (network) =
  --   let layer = dimension.from_2d_1d
  --   in add_layer layer network
  --
  -- def from_3d_1d (network) =
  --   let (l, m, n) = network.shape
  --   let layer = dimension.from_3d_1d (l * m * n)
  --   in add_layer layer network

  def forward 'input_type 'all_shapes 'output 'cw 'rw (input: input_type) (network: nn_type all_shapes input_type output cw rw) =
    let { weights, forward, seed = _, shape = _ } = network
    in forward input weights

}
