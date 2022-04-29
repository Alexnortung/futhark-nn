type shape_1d = i64
type shape_2d = (i64, i64)
type shape_3d = (i64, i64, i64)

-- optimizer general types
type^ optimizer_apply_optimize_func 'wb = wb -> wb -> wb -- the optimizer's apply function, which will return the new weights
type^ optimizer_apply_record 't = {
  apply_1d: optimizer_apply_optimize_func ([]t),
  apply_2d: optimizer_apply_optimize_func ([][]t),
  apply_3d: optimizer_apply_optimize_func ([][][]t),
  apply_4d: optimizer_apply_optimize_func ([][][][]t)
}
type^ optimizer_loss_function 'input 'output 'weights 't = input -> output -> weights -> t -- the loss function for the optimizer

-- types from layer.init
type^ layer_fwd_type 'options 'layer_input 'wb 'out = options -> layer_input -> wb -> out
type^ layer_apply_optimize_type 't 'options 'weights = options -> optimizer_apply_record t -> weights -> weights -> weights
type^ layer_type 't 'options 'layer_input 'wb 'shape 'out = {
  forward: layer_fwd_type options layer_input wb out,
  apply_optimize: layer_apply_optimize_type t options wb,
  options: options,
  weights: wb,
  shape: shape
}

-- from nn
type^ nn_type 't 'shape_type 'input 'output 'current_weight 'rest_weights = {
  seed: i32,
  shape: shape_type,
  forward: input -> (current_weight, rest_weights) -> output,
  apply_optimize: (optimizer_apply_record t) -> (current_weight, rest_weights) -> (current_weight, rest_weights) -> (current_weight, rest_weights),
  weights: (current_weight, rest_weights)
}

-- optimizer
type^ optimizer_type 't 'options 'input 'output 'current_weight 'rest_weight = {
  options: options,
  apply_gradient: optimizer_apply_record t,
  loss_function: optimizer_loss_function input output (current_weight, rest_weight) t
}
