import "../lib/github.com/diku-dk/linalg/linalg"

module fully_connected_simple (R:real) = {
  type t = R.t

  module lalg   = mk_linalg R

  let forward [m] [n]
    (input: [m]t) -- the values of the input nodes
    (weights: [n][m]t)
    (biases: [n]t)
    (activation_func: t -> t) : [n]t =
      let propagated = lalg.matvecmul_row weights input
      let biased = map2 (\x b -> R.(x + b)) propagated biases
      --let biased = map2 (\xrow b -> map (\x -> R.(x + b)) xrow) propagated biases
      -- apply the activation function on each output node
      let activated = map (\x -> activation_func x) biased
      in activated
}
