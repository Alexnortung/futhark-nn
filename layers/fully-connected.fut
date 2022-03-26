import "../lib/github.com/diku-dk/linalg/linalg"

module fully_connected_simple (R:real) = {
  type t = R.t

  module lalg   = mk_linalg R

  let forward [m] [n] -- m input nodes and n output nodes
    (input: [m]t) -- the values of the input nodes
    (activation_func: t -> t)
    (weights: [n][m]t)
    (biases: [n]t)
    : [n]t =
      let propagated = lalg.matvecmul_row weights input
      let biased = map2 (\x b -> R.(x + b)) propagated biases
      --let biased = map2 (\xrow b -> map (\x -> R.(x + b)) xrow) propagated biases
      -- apply the activation function on each output node
      let activated = map (\x -> activation_func x) biased
      in activated

  let backward [m] [n]
    (input: [m]t)
    (activation_func: t -> t)
    (weights: [n][m]t)
    (biases: [n]t)
    (error: [n]t)
    (alpha: t) -- learning rate
    (is_first_layer: bool)
    : ([n]t, [n][m]t, [n]t) =
      let grad = vjp (forward input activation_func)
      let vector = grad weights biases
      let learning_vector = map (\x -> x * alpha) vector


}
