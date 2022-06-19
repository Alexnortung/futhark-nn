import "../layers/convolutional"
import "../layers/linear"
import "../layers/pooling"
import "../layers/dimension"
import "../layers/types"
import "../optimizers/optimizers"
import "../util/loss-func"
import "../util/activation-func"

module neural_network (R:real) = {
  type t = R.t

  type^ nn_type 'shape_type 'input 'output 'current_weight 'rest_weights = nn_type t shape_type input output current_weight rest_weights

  module layers = {
    module convolutional = convolutional R
    module linear = linear R
    module maxpool = pooling R
    module dimension = dimension R
  }

  
  module optim = optimizers R

  module activation = activation_func R
  module loss = loss R

  def compose_forward [k] 'input_type 'prev_wbs 'layer_input 'current_wbs 'output
    (prev_forward: [k]input_type -> prev_wbs -> [k]layer_input)
    (layer_forward: [k]layer_input -> current_wbs -> [k]output)
    (input: [k]input_type)
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


  def init_forward 'input_type (k: i64) (x: [k]input_type) _ : [k]input_type = x
  def init_apply_optimize (_apply) (w) (_wg) = w

  def init_1d (n: i64) (seed: i32) : nn_type shape_1d ([n]t) ([n]t) () () =
    {
      seed,
      shape = n,
      forward = init_forward,
      apply_optimize = init_apply_optimize,
      weights = ((), ())
    }

  def init_2d (m: i64) (n: i64) (seed: i32) : nn_type shape_2d ([m][n]t) ([m][n]t) () () =
    {
      seed,
      shape = (m, n),
      forward = init_forward,
      apply_optimize = init_apply_optimize,
      weights = ((), ())
    }

  def init_3d (l: i64) (m: i64) (n: i64) (seed: i32) : nn_type shape_3d ([l][m][n]t) ([l][m][n]t) () () =
    {
      seed,
      shape = (l, m, n),
      forward = init_forward,
      apply_optimize = init_apply_optimize,
      weights = ((), ())
    }

  def init_4d (o: i64) (l: i64) (m: i64) (n: i64) (seed: i32) : nn_type shape_4d ([o][l][m][n]t) ([o][l][m][n]t) () () =
    {
      seed,
      shape = (o, l, m, n),
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
    let new_forward = (\k -> compose_forward (forward k) (layer_forward k layer_options))
    let new_apply_optimize = compose_apply_optimize apply_optimize (layer_apply_optimize layer_options)
    in {
      seed,
      shape = layer_shape,
      weights = (layer_weights, weights),
      apply_optimize = new_apply_optimize,
      forward = new_forward
    }

  def conv_2d 'input_type 'prev_current_weight 'prev_rest_weight 'current_weight
    [prev_m] [prev_n] [in_channels]
    (output_m: i64)
    (output_n: i64)
    (out_channels: i64)
    (kernel_m: i64)
    (kernel_n: i64)
    (activation_func: activation_type t)
    (network: nn_type shape_3d input_type ([in_channels][prev_m][prev_n]t) prev_current_weight prev_rest_weight)
    : nn_type shape_3d input_type ([out_channels][output_m][output_n]t) (layers.convolutional.weight_bias_2d [in_channels] [out_channels] [kernel_m] [kernel_n]) (prev_current_weight, prev_rest_weight)
    =
      let seed = network.seed
      let layer = layers.convolutional.init_2d output_m output_n in_channels out_channels kernel_m kernel_n activation_func seed
      in add_layer layer network

  def linear 'rest_weights 'input_type 'prev_current_weight 'prev_rest_weight
    [m]
    (n: i64)
    (activation: activation_type t)
    (network: nn_type shape_1d input_type ([m]t) prev_current_weight prev_rest_weight)
    : nn_type shape_1d input_type ([n]t) (layers.linear.weights_and_bias [m] [n]) (prev_current_weight, prev_rest_weight) =
      let seed = network.seed
      let layer = layers.linear.init m n activation seed
      in add_layer layer network

  def maxpool_2d 'input_type  'prev_current_weight 'prev_rest_weight 'current_weight
    [prev_m] [prev_n]
    (output_m: i64)
    (output_n: i64)
    (network: nn_type shape_2d input_type ([prev_m][prev_n]t) prev_current_weight prev_rest_weight)
    : nn_type shape_2d input_type ([output_m][output_n]t) () (prev_current_weight, prev_rest_weight)
    =
      let layer = layers.maxpool.init_2d output_m output_n
      in add_layer layer network

  def from_1d_2d (output_m) (output_n) (network) =
    let layer = layers.dimension.from_1d_2d output_m output_n
    in add_layer layer network

  def from_1d_3d (output_l) (output_m) (output_n) (network) =
    let layer = layers.dimension.from_1d_3d output_l output_m output_n
    in add_layer layer network

  def from_2d_1d (network) =
    let (m, n) = network.shape
    let layer = layers.dimension.from_2d_1d (m * n)
    in add_layer layer network

  def from_3d_1d (network) =
    let (l, m, n) = network.shape
    let layer = layers.dimension.from_3d_1d (l * m * n)
    in add_layer layer network

  def set_weights (new_weights) (network) =
    let { seed, shape, forward, weights = _, apply_optimize } = network
    in {
      seed, shape, forward, apply_optimize,
      weights = new_weights
    }

  def forward 'input_type 'all_shapes 'output 'cw 'rw [k] (input: [k]input_type) (network: nn_type all_shapes input_type output cw rw) =
    let { weights, forward, apply_optimize = _, seed = _, shape = _ } = network
    in forward k input weights

  def make_loss [k] [n] 'input_type 'cw 'rw
    (loss_function: (sz: i64) -> [sz]t -> [sz]t -> t)
    (network: nn_type shape_1d (input_type) ([n]t) cw rw)
    (input: [k]input_type) (expected: [k][n]t) (weights: (cw, rw))
    : t =
      let forward = network.forward
      let output = forward k input weights
      let kn = k * n
      let output = flatten_to kn output
      let loss = loss_function kn output (flatten_to kn expected)
      in loss

  def argmax [n] (input: [n]t) : i64 =
    reduce (\acc_i i -> if R.(input[acc_i] > input[i]) then acc_i else i) 0 (iota n)

  -- TODO: make more generic
  def accuracy 'cw 'rw 'input_type [k] [n]
    (input: [k]input_type)
    (labels: [k][n]t)
    (_activation)
    (network: nn_type shape_1d (input_type) ([n]t) cw rw)
    : t =
      -- TODO: make more generic
      let labels = map (argmax) labels
      -- get the output from forward propagation
      let output = forward input network
      -- TODO: apply actiation
      -- find predicted values
      let predictions = map (argmax) output
      -- find the number of hits
      let hits_array = map2 (\prediction label -> i64.bool (prediction == label)) predictions labels
      let hits = i64.sum hits_array
      in R.(i64 hits / i64 n)


  def train [k] 'shape 'input_type 'output_type 'cw 'rw 'o_options
    (input: [k]input_type)
    (output: [k]output_type) -- aka labels
    (iterations: i64)
    (optimizer: optimizer_type t o_options ([k]input_type) ([k]output_type) cw rw)
    (network: nn_type shape input_type output_type cw rw)
    : nn_type shape input_type output_type cw rw =
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

  -- sets the weight on the latest layer
  def change_weight 'shape 'input 'output 'cw 'rw (new_weights: cw) (network: nn_type shape input output cw rw) =
    let { shape, apply_optimize, weights, forward, seed } = network
    let (_, rest_weights) = weights
    let new_network_weights = (new_weights, rest_weights)
    in { shape, apply_optimize, weights = new_network_weights, forward, seed }
}

-- TESTS

local module nn_test = neural_network f64

-- ==
-- entry: nn_linear_test
-- input {
--    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
--    
--    1i32
-- }
-- output {
--    [[-18.80553453580382f64]]
-- }
entry nn_linear_test [k] (input: [k][]f64) (seed: i32) =
  nn_test.init_1d 7 seed
  |> nn_test.linear 5 (nn_test.activation.identity)
  |> nn_test.linear 6 (nn_test.activation.identity)
  |> nn_test.linear 1 (nn_test.activation.identity)
  |> nn_test.forward input

-- ==
-- entry: nn_conv_test
-- input {
--  [[[[1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0]]],
--   [[[1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0]]]]
--
--    1i32
-- }
-- auto output

entry nn_conv_test [k] (input: [k][1][6][5]f64) (seed: i32) =
  nn_test.init_3d 1 6 5 seed
  |> nn_test.conv_2d 4 4 1 3 2 (nn_test.activation.identity)
  |> nn_test.conv_2d 2 3 1 3 2 (nn_test.activation.identity)
  |> nn_test.forward input

-- ==
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
--    [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]
--  ]
-- }
-- output { [
--   [[2.0, 4.0, 2.0, 4.0],
--    [2.0, 4.0, 2.0, 4.0]]
-- ] }
entry nn_maxpool_2d_test (input: [][][]f64) =
  nn_test.init_2d 8 8 1
  |> nn_test.maxpool_2d 4 4
  |> nn_test.maxpool_2d 2 4
  |> nn_test.forward input

-- ==
-- entry: nn_add_layer_test
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
--          [ 93.0, 109.0]]]
--         ]
-- }

entry nn_add_layer_test (input: [][][][]f64 ) (b1: []f64) (w1: [][][][]f64) =
  let seed = 1
  in nn_test.init_3d 1 3 3 seed
  |> nn_test.add_layer (nn_test.layers.convolutional.init_2d 2 2 1 1 2 2 (nn_test.activation.identity) seed
                   |> nn_test.layers.convolutional.set_weights w1
                   |> nn_test.layers.convolutional.set_bias b1)
  |> nn_test.forward input

