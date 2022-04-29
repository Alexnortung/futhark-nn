import "../util/linalg"
import "../layers/types"

module sgd (R: real) = {
  type t = R.t
  type weights_bias [m] [n] [bn] = ([m][n]t, [bn]t)
  type options = {
    learning_rate: t
  }

  type^ sgd_optimizer 'shape 'input 'output 'cw 'rw = optimizer_type t options input output cw rw

  module lalg = mk_linalg R

  def apply_1d [n]
    (learning_rate: t)
    (w: [n]t)
    (wg: [n]t) -- gradient values
    : [n]t =
      -- scale the gradient weight and bias to the learning rate
      let wg_scaled = lalg.vecmul_scalar wg learning_rate
      -- apply gradient descent by subtracting the gradient
      in lalg.vecsub w wg_scaled

  def apply_2d [m][n]
    (learning_rate: t)
    (w: [m][n]t)
    (wg: [m][n]t) -- gradient values
    : [m][n]t =
      -- scale the gradient weight and bias to the learning rate
      let wg_scaled = lalg.matmul_scalar wg learning_rate
      -- apply gradient descent by subtracting the gradient
      in lalg.matsub w wg_scaled

  def apply_3d
    (learning_rate: t)
    (w)
    (wg) =
      map2 (\w_2d wg_2d ->
        apply_2d learning_rate w_2d wg_2d
      ) w wg

  def apply_4d
    (learning_rate: t)
    (w)
    (wg) =
      map2 (\w wg ->
        apply_3d learning_rate w wg
      ) w wg

  def init 'input 'output 'weights (learning_rate: t) (loss_function: input -> output -> weights -> t) =
    {
      options = {
        learning_rate
      },
      apply_gradient = {
        apply_1d = apply_1d learning_rate,
        apply_2d = apply_2d learning_rate,
        apply_3d = apply_3d learning_rate,
        apply_4d = apply_4d learning_rate
      },
      loss_function
    }

}

