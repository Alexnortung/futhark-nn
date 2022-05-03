-- import "../layers/convolutional"
import "../layers/linear"
-- import "../layers/pooling"
-- import "../layers/dimension"
import "../layers/types"
import "../optimizers/optimizers"
import "../util/loss-func"

module neural_network (R:real) = {
  type t = R.t

  type^ nn_type 'shape_type 'input 'output 'current_weight 'rest_weights = nn_type t shape_type input output current_weight rest_weights

  module layers = {
    module convolutional = convolutional R
    module linear = linear R
    module maxpool = pooling R
    module dimension = dimension
  }

  
  module optim = optimizers R

  module activation = import "../util/activation-func"
  module loss = loss R

  def compose_forward 'input_type 'prev_wbs 'layer_input 'current_wbs 'output
    (prev_forward: input_type -> prev_wbs -> layer_input)
    (layer_forward: layer_input -> current_wbs -> output)
    (input: input_type)
    (wbs: (current_wbs, prev_wbs)) =
      let (current_wb, prev_wbs) = wbs
      let prev_result = prev_forward input prev_wbs
      in layer_forward prev_result current_wb

  def compose_apply_optimize 'current_wb 'rest_wbs
    (prev_apply: optimizer_apply_record t -> rest_wbs -> rest_wbs -> rest_wbs)
    (layer_apply: optimizer_apply_record t -> current_wb -> current_wb -> current_wb)
    (apply_record: optimizer_apply_record t)
    (weights: (current_wb, rest_wbs))
    (gradient_weights: (current_wb, rest_wbs)) : (current_wb, rest_wbs) =
      let (current_weights, rest_weights) = weights
      let (current_gradient, rest_gradient) = gradient_weights
      let prev_applied = prev_apply apply_record rest_weights rest_gradient
      let current_applied = layer_apply apply_record current_weights current_gradient
      in (current_applied, prev_applied)


  def init_forward (x) _ = x
  def init_apply_optimize (_apply) (w) (_wg) = w

  def init_1d [k] (n: i64) (seed: i32) : nn_type shape_1d ([k][n]t) ([k][n]t) () () =
    {
      seed,
      shape = n,
      forward = init_forward,
      apply_optimize = init_apply_optimize,
      weights = ((), ())
    }

  def init_2d [k] (m: i64) (n: i64) (seed: i32) : nn_type shape_2d ([k][m][n]t) ([k][m][n]t) () () =
    {
      seed,
      shape = (m, n),
      forward = init_forward,
      apply_optimize = init_apply_optimize,
      weights = ((), ())
    }

  def add_layer 'nn_input 'l_input 'l_options  'l_wb 'l_shape 'l_out 'nn_shape 'prev_current_weight 'prev_rest_weight
    (layer: layer_type t l_options l_input l_wb l_shape l_out)
    (network: nn_type nn_shape nn_input l_input prev_current_weight prev_rest_weight)
    : nn_type l_shape nn_input l_out l_wb (prev_current_weight, prev_rest_weight) =
    let {
      apply_optimize = layer_apply_optimize,
      forward = layer_forward,
      options = layer_options,
      weights = layer_weights,
      shape = layer_shape
    } = layer
    let { shape = _, apply_optimize, weights, forward, seed } = network
    let new_forward = compose_forward forward (layer_forward layer_options)
    let new_apply_optimize = compose_apply_optimize apply_optimize (layer_apply_optimize layer_options)
    in {
      seed,
      shape = layer_shape,
      weights = (layer_weights, weights),
      apply_optimize = new_apply_optimize,
      forward = new_forward
    }

  -- def conv_2d_shape 'input 'output 'cw 'rw (kernel_m: i64) (kernel_n: i64) (network: nn_type shape_2d input output cw rw) : shape_2d =
  --   let shape = network.shape
  --   let output_m = shape.0 - kernel_m + 1
  --   let output_n = shape.1 - kernel_n + 1
  --   in (output_m, output_n)
  --
  -- def conv_2d 'input_type  'prev_current_weight 'prev_rest_weight 'current_weight
  --   [k] [prev_m] [prev_n]
  --   (output_m: i64)
  --   (output_n: i64)
  --   (kernel_m: i64)
  --   (kernel_n: i64)
  --   (network: nn_type shape_2d input_type ([k][prev_m][prev_n]t) prev_current_weight prev_rest_weight)
  --   : nn_type shape_2d input_type ([k][output_m][output_n]t) ([kernel_m][kernel_n]t, t) (prev_current_weight, prev_rest_weight)
  --   =
  --     let { shape = _, weights, forward, seed } = network
  --     let layer = conv.init_2d output_m output_n kernel_m kernel_n seed
  --     let { forward = layer_forward, options = layer_options, weights = layer_weights, shape = _} = layer
  --     let new_forward = compose_forward forward (layer_forward layer_options)
  --     in {
  --       seed,
  --       shape = (output_m, output_n),
  --       weights = (layer_weights, weights),
  --       forward = new_forward
  --     }

  def linear 'rest_weights 'input_type 'prev_current_weight 'prev_rest_weight
    [k] [m]
    -- (m: i64)
    (n: i64)
    (activation: t -> t)
    (network: nn_type shape_1d input_type (layers.linear.input_type [k] [m]) prev_current_weight prev_rest_weight)
    : nn_type shape_1d input_type (layers.linear.output_type [k] [n]) (layers.linear.weights_and_bias [m] [n]) (prev_current_weight, prev_rest_weight) =
      let seed = network.seed
      let layer = layers.linear.init m n activation seed
      in add_layer layer network

  -- def maxpool_2d_shape 'input 'output 'cw 'rw (window_m: i64) (window_n: i64) (network: nn_type shape_2d input output cw rw) : shape_2d =
  --   let (m, n) = network.shape
  --   in (m / window_m, n / window_n)
  --
  -- def maxpool_2d 'input_type  'prev_current_weight 'prev_rest_weight 'current_weight
  --   [k] [prev_m] [prev_n]
  --   (output_m: i64)
  --   (output_n: i64)
  --   (network: nn_type shape_2d input_type ([k][prev_m][prev_n]t) prev_current_weight prev_rest_weight)
  --   : nn_type shape_2d input_type ([k][output_m][output_n]t) () (prev_current_weight, prev_rest_weight)
  --   =
  --     let { shape = _, weights, forward, seed } = network
  --     let layer = mpool.init_2d output_m output_n
  --     let { forward = layer_forward, options = layer_options, weights = layer_weights, shape = _ } = layer
  --     let new_forward = compose_forward forward (layer_forward layer_options)
  --     in {
  --       seed,
  --       shape = (output_m, output_n),
  --       weights = (layer_weights, weights),
  --       forward = new_forward
  --     }

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

  def set_weights (new_weights) (network) =
    let { seed, shape, forward, weights = _, apply_optimize } = network
    in {
      seed, shape, forward, apply_optimize,
      weights = new_weights
    }

  def forward 'input_type 'all_shapes 'output 'cw 'rw (input: input_type) (network: nn_type all_shapes input_type output cw rw) =
    let { weights, forward, apply_optimize = _, seed = _, shape = _ } = network
    in forward input weights

  def make_loss [k] [n1] [n] 'cw 'rw
    (loss_function: (sz: i64) -> [sz]t -> [sz]t -> t)
    (network: nn_type shape_1d ([k][n1]t) ([k][n]t) cw rw)
    (input: ([k][n1]t)) (expected: [k][n]t) (weights: (cw, rw))
    : t =
      let forward = network.forward
      let output = forward input weights
      let kn = k * n
      let output = flatten_to kn output
      let loss = loss_function kn output (flatten_to kn expected)
      in loss

  def train 'shape 'input 'output_type 'cw 'rw 'o_options
    (input: input)
    (output: output_type)
    (iterations: i64)
    (optimizer: optimizer_type t o_options input output_type cw rw)
    (network: nn_type shape input output_type cw rw)
    : nn_type shape input output_type cw rw =
      let { loss_function, options = _, apply_gradient } = optimizer
      let initial_weights = network.weights
      let loss = loss_function input output
      let gradient_func = vjp loss
      let optimized_weights = loop nn_weights = initial_weights for _i < iterations do
        let gradient = gradient_func nn_weights R.(i64 1)
        let new_weights = network.apply_optimize apply_gradient nn_weights gradient
        in new_weights
      let new_network = set_weights optimized_weights network
      in new_network
}
