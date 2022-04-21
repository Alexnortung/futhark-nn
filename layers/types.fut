-- types from layer.init
type^ layer_fwd_type 'options 'layer_input 'wb '~out = options -> layer_input -> wb -> out
type^ layer_type 'options 'layer_input 'wb '~out = {
  forward: layer_fwd_type options layer_input wb out,
  options: options,
  weights: wb
}
