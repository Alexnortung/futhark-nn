type shape_1d = i64
type shape_2d = (i64, i64)
type shape_3d = (i64, i64, i64)

-- types from layer.init
type^ layer_fwd_type 'options 'layer_input 'wb 'out = options -> layer_input -> wb -> out
type^ layer_type 'options 'layer_input 'wb 'shape 'out = {
  forward: layer_fwd_type options layer_input wb out,
  options: options,
  weights: wb,
  shape: shape
}

-- from nn
type^ nn_type 'shape_type 'input 'output 'current_weight 'rest_weights = {
  seed: i32,
  shape: shape_type,
  forward: input -> (current_weight, rest_weights) -> output,
  weights: (current_weight, rest_weights)
}
