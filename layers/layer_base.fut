import "types"

module layer_base = {
  def forward_layer [k] 't 'input_type 'options 'wb 'shape 'output_type
    (input: [k]input_type)
    (layer: layer_type t options (input_type) wb shape (output_type))
    : [k]output_type =
      let { forward, apply_optimize = _, options, weights, shape = _ } = layer
      in forward k options input weights
}
