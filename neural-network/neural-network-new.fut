import "../layers/convolutional"

module neural_network (R:real) = {
  type t = R.t
  type shape_1d = (i64)
  type shape_2d = (i64, i64)
  type shape_3d = (i64, i64, i64)
  type^ nn_type 'shape_type 'input 'output 'current_weight 'rest_weights = {
    seed: i32,
    shape: shape_type,
    forward: input -> (current_weight, rest_weights) -> output,
    weights: (current_weight, rest_weights)
  }

  module conv = convolutional R

  def compose_forward 'input_type 'prev_wbs 'layer_input 'current_wbs 'output
    (prev_forward: input_type -> prev_wbs -> layer_input)
    (layer_forward: layer_input -> current_wbs -> output)
    (input: input_type)
    (wbs: (current_wbs, prev_wbs)) =
      let (current_wb, prev_wbs) = wbs
      let prev_result = prev_forward input prev_wbs
      in layer_forward prev_result current_wb

  def init_2d [k] (m: i64) (n: i64) (seed: i32) : nn_type shape_2d ([k][m][n]t) ([k][m][n]t) () () =
    {
      seed,
      shape = (m, n),
      forward = (\x _ -> x),
      weights = ((), ())
    }

  def conv_2d_shape 'input 'output 'cw 'rw (kernel_m: i64) (kernel_n: i64) (network: nn_type shape_2d input output cw rw) : shape_2d =
    let shape = network.shape
    let output_m = shape.0 - kernel_m + 1
    let output_n = shape.1 - kernel_n + 1
    in (output_m, output_n)

  def conv_2d 'input_type  'prev_current_weight 'prev_rest_weight 'current_weight
    [k] [prev_m] [prev_n]
    (output_m: i64)
    (output_n: i64)
    (kernel_m: i64)
    (kernel_n: i64)
    (network: nn_type shape_2d input_type ([k][prev_m][prev_n]t) prev_current_weight prev_rest_weight)
    : nn_type shape_2d input_type ([k][output_m][output_n]t) ([kernel_m][kernel_n]t, t) (prev_current_weight, prev_rest_weight)
    =
      let { shape = _, weights, forward, seed } = network
      let layer = conv.init_2d_known output_m output_n kernel_m kernel_n seed
      let (layer_forward, layer_options, layer_weights) = layer
      let new_forward = compose_forward forward (layer_forward layer_options)
      in {
        seed,
        shape = (output_m, output_n),
        weights = (layer_weights, weights),
        forward = new_forward
      }

  def forward 'input_type 'all_shapes 'output 'cw 'rw (input: input_type) (network: nn_type all_shapes input_type output cw rw) =
    let { weights, forward, seed = _, shape = _ } = network
    in forward input weights

}

-- local def model [k] [m] [n] 't (state) (input: [k][m][n]t) (weights) =
--   nn.init_2d state input weights
--   |> nn.conv_2d 3 2
--   -- |> nn.maxpool_2d 2 2
--   -- |> nn.from_2d_to_1d
--   -- |> nn.linear 1
--
-- local entry simple_test (input) =
--   let weights = neural_network.gen_weights model input
--
