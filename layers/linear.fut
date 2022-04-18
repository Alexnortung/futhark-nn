import "../lib/github.com/diku-dk/linalg/linalg"
import "../util/weight-initialization"
import "types"

module linear (R:real) = {
  type t = R.t
  type weights_type [m] [n] = [n][m]t
  type input_type [k] [m] = [k][m]t
  type output_type [k] [n] = [k][n]t
  type bias_type [n] = [n]t
  type weights_and_bias [m] [n] = (weights_type [m] [n], bias_type [n])
  -- type layer_type [k] [m] [n] ((weights_and_bias m n) -> [k][n]t, weights_and_bias m n)
  type^ linear_layer_fwd [k] [m] [n] = layer_fwd_type () (input_type [k] [m]) (weights_and_bias [m] [n]) (output_type [k] [n])
  type^ linear_layer_type [k] [m] [n] = layer_type () (input_type [k] [m]) (weights_and_bias [m] [n]) (output_type [k] [n])

  module lalg = mk_linalg R

  module wi = weight_init R

  let forward  [k] [m] [n] -- k batches, m input nodes and n output nodes
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

  let forward_layer [k] [m] [n] (layer: linear_layer_type [k] [m] [n]) (input: input_type [k] [m]) : output_type [k] [n] =
    -- take the forward function (layer.0) and apply the input and the weights + bias (layer.1)
    let (function, options, wb) = layer
    let output = function options input wb
    in output

  let backward [k] [m] [n]
    (forward_weights: linear_layer_fwd [k] [m] [n])
    (learning_rate: t)
    ((current_weights, current_bias): weights_and_bias [m] [n])
    ((gradient_weights, gradient_bias): weights_and_bias [m] [n])
    : linear_layer_type [k] [m] [n] =
      let new_weights = map2 (\cw gw ->
        map2 (\cw gw ->
          R.(cw - learning_rate * gw)
        ) cw gw
      ) current_weights gradient_weights

      let new_bias = map2 (\cb gb ->
        R.(cb - learning_rate * gb)
      ) current_bias gradient_bias
      in (forward_weights, (), (new_weights, new_bias))
    

  let init [k] (m: i64) (n: i64) (activation_func: t -> t) (seed: i32) : linear_layer_type [k] [m] [n] =
    let weights = wi.gen_2d m n seed
    let biases = wi.gen_1d n seed
    -- make a function that represents the forward function, but only needs an input
    let forward_weights = (\_ input (weights, biases) -> forward input activation_func weights biases)
    in (forward_weights, (), (weights, biases))

  let set_weights [k] [m] [n] (layer: linear_layer_type [k] [m] [n]) (new_weights: weights_type [m] [n]) : linear_layer_type [k] [m] [n] =
    let (fwd, options, (_, biases)) = layer
    let new_layer = (fwd, options, (new_weights, biases))
    in new_layer

  let set_bias [k] [m] [n] (layer: linear_layer_type [k] [m] [n]) (new_bias: bias_type [n]) : linear_layer_type [k] [m] [n] =
    let (fwd, options,  (weights, _)) = layer
    let new_layer = (fwd, options, (weights, new_bias))
    in new_layer
}
