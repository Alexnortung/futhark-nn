import "../lib/github.com/diku-dk/linalg/linalg"

module fully_connected = {
  type t = f64

  module lalg   = mk_linalg t

  let forward [m] [n]
    (input: [m]t) -- the values of the input nodes
    (weights: [n][m]t)
    (biases: [n][m]t)
    (activation_func: t -> t) : [n]t =
      let propagated = lalg.matmul (tranpose input) weights
      let biased = map2 (\x b' -> x + b') propagated biases
      -- apply the activation function on each output node
      let activated = map (\x -> activation_func x) biased
      in activated
}
