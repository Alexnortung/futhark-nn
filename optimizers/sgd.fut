import "../util/linalg"
import "../neural-network/neural-network-new"
import "../layers/types"

module sgd (R: real) = {
  local module nn = neural_network R

  type t = R.t
  type weights_bias [m] [n] [bn] = ([m][n]t, [bn]t)
  type options = {
    learning_rate: t
  }

  type sgd_optimizer 'shape 'input 'output 'cw 'rw = optimizer_type t options shape input output cw rw

  module lalg = mk_linalg R

  def init 'input 'output 'weights (learning_rate: t) (loss_function: input -> output -> weights -> t) (network) =
    {
      options: {
        learning_rate
      },
      network,
      loss_function
    }

  def apply [m] [n] [bn]
    (learning_rate: t)
    ((w, b): weights_bias [m] [n] [bn])
    ((wg, bg): weights_bias [m] [n] [bn])
    : weights_bias [m] [n] [bn] =
      -- scale the gradient weight and bias to the learning rate
      let wg_scaled = matmul_scalar wg learning_rate
      let bg_scaled = matmul_scalar bg learning_rate

      -- apply gradient descent by subtracting the gradient
      let new_weights = matsub w wg_scaled
      let new_bias = vecsub b bg_scaled
      in (new_weights, new_bias)

  def train 'shape 'input 'output 'cw 'rw
    (input: input)
    (output: output)
    (iterations: i64)
    (optimizer: sgd_optimizer shape input output cw rw)
    : sgd_optimizer shape input output cw rw =
      let { network, loss_function, options } = optimizer
      let { learning_rate } = options
      let initial_weights = network.weights
      let loss = loss_function input
      let apply_gradient = apply learning_rate
      let optimized_weights = loop nn_weights = initial_weights for i < iterations do
        let gradient = vjp loss nn_weights 1
        let new_weights = nn.backward apply_gradient nn_weights gradient network
        in new_weights
      let new_network = nn.set_weights optimized_weights
      in {
        options,
        loss_function,
        network = new_network
      }
}

