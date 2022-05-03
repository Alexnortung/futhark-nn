-- import "../lib/github.com/diku-dk/linalg/linalg"
import "../util/linalg"
import "../util/weight-initialization"
import "types"

module linear (R:real) = {
  type t = R.t
  type weights_type [m] [n] = [n][m]t
  type input_type [k] [m] = [k][m]t
  type output_type [k] [n] = [k][n]t
  type bias_type [n] = [n]t
  type weights_and_bias [m] [n] = (weights_type [m] [n], bias_type [n])
  type options = ()
  -- type layer_type [k] [m] [n] ((weights_and_bias m n) -> [k][n]t, weights_and_bias m n)
  type^ linear_layer_fwd [k] [m] [n] = layer_fwd_type () (input_type [k] [m]) (weights_and_bias [m] [n]) (output_type [k] [n])
  type^ linear_layer_type [k] [m] [n] = layer_type t options (input_type [k] [m]) (weights_and_bias [m] [n]) (i64) (output_type [k] [n])

  module lalg = mk_linalg R

  module wi = weight_init R

  def forward  [k] [m] [n] -- k batches, m input nodes and n output nodes
    (input: input_type [k] [m]) -- the values of the input nodes
    (activation_func: t -> t)
    (weights: weights_type [m] [n])
    (biases: bias_type [n])
    : output_type [k] [n] =
      -- TODO: use matmul instead of map matvecmul_row
      map (\input ->
        let propagated = lalg.matvecmul_row weights input
        let biased = map2 (\x b -> R.(x + b)) propagated biases
        --let biased = map2 (\xrow b -> map (\x -> R.(x + b)) xrow) propagated biases
        -- apply the activation function on each output node
        let activated = map (\x -> activation_func x) biased
        in activated
      ) input

  def forward_layer [k] [m] [n] (input: input_type [k] [m]) (layer: linear_layer_type [k] [m] [n]) : output_type [k] [n] =
    -- take the forward function (layer.0) and apply the input and the weights + bias (layer.1)
    let { forward, options, weights, apply_optimize = _, shape = _ } = layer
    let output = forward options input weights
    in output

  def apply_optimize [m] [n]
    (_options: options)
    (apply_func_record: optimizer_apply_record t)
    ((current_weights, current_bias): weights_and_bias [m] [n])
    ((gradient_weights, gradient_bias): weights_and_bias [m] [n])
    : weights_and_bias [m] [n] =
      let new_weights = apply_func_record.apply_2d n m current_weights gradient_weights
      let new_bias = apply_func_record.apply_1d n current_bias gradient_bias
      in (new_weights, new_bias)
    

  def init [k] (m: i64) (n: i64) (activation_func: t -> t) (seed: i32) : linear_layer_type [k] [m] [n] =
    let weights = wi.gen_2d n m seed
    let biases = wi.gen_1d n seed
    -- make a function that represents the forward function, but only needs an input
    let forward_weights = (\_ input (weights, biases) -> forward input activation_func weights biases)
    in {
      apply_optimize,
      forward = forward_weights,
      weights = (weights, biases),
      options = (),
      shape = n
    }

  def set_weights [k] [m] [n] (new_weights: weights_type [m] [n]) (layer: linear_layer_type [k] [m] [n]) : linear_layer_type [k] [m] [n] =
    let { forward, apply_optimize, options, shape, weights = (_, biases) } = layer
    let new_layer = { apply_optimize, forward, options, shape, weights = (new_weights, biases) }
    in new_layer

  def set_bias [k] [m] [n] (new_bias: bias_type [n]) (layer: linear_layer_type [k] [m] [n]) : linear_layer_type [k] [m] [n] =
    let { forward, apply_optimize, options, shape, weights = (weights, _) } = layer
    let new_layer = { apply_optimize, forward, options, shape, weights = (weights, new_bias) }
    in new_layer
}

-- TESTS

local module test_linear = linear f64

-- ==
-- entry: linear_forward linear_init_forward
-- input {[
--        [1.0, 2.0, 3.0, 4.0],
--        [2.0, 3.0, 4.0, 5.0],
--        [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[
--          [ 31.0,  72.0, 113.0],
--          [ 41.0,  98.0, 155.0],
--          [ 51.0, 124.0, 197.0]
--         ]}

entry linear_forward [k] (input: [k][]f64) w b : [k][]f64 =
  let output = test_linear.forward input (\x -> x) w b
  in output

entry linear_init_forward [k] [m] [n] (input: [k][m]f64) (w: [n][m]f64) (b: [n]f64) : [k][n]f64 =
  test_linear.init m n (\x -> x) 1
  |> test_linear.set_weights w
  |> test_linear.set_bias b
  |> test_linear.forward_layer input

