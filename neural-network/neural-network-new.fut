import "../layers/convolutional"

module neural_network (R:real) = {
  type t = R.t
  module conv = convolutional t

  def init_2d [k] [m] [n] 't (state) (input: [k][m][n]t) (weights) =
    {
      shape = (m, n),
      weights = weights,
      result = input,
    }

  def conv_2d (kernel_m) (kernel_n) (inter) =
    let { shape, weights, result = input} = inter
    let output_m = shape.0 - kernel_m + 1
    let output_n = shape.1 - kernel_n + 1
    let (current_weights, next_weights) = weights
    let (weights, bias) = current_weights
    let result convolutional.forward_2d_known output_m output_n input bias current_weights
    in {
      shape = (output_m, output_n),
      weights = next_weights,
      result,
    }

}

local def model [k] [m] [n] 't (state) (input: [k][m][n]t) (weights) =
  nn.init_2d state input weights
  |> nn.conv_2d 3 2
  -- |> nn.maxpool_2d 2 2
  -- |> nn.from_2d_to_1d
  -- |> nn.linear 1

local entry simple_test (input) =
  let weights = neural_network.gen_weights model input

