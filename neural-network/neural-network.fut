import "../layers/types"

module neural_network = {
  -- acutally recursive type, but futhark does not support this
  type nn_type 'fi 'fw 'fo 't 'nn_rest = ((fi -> fw -> fo, t), nn_rest)

  let init () =
    -- construct the base for the neural network.
    -- this part is basically the identity function, 
    -- but allows for a second argument have a consistent interface
    (((\input _ -> input), ()), ())

  let add_layer (nn: nn_type) (layer: layer_type) =
    -- the layer should be a layer gotten from layer_module.init
    -- get the layer's forward function and the initial weights
    --
    -- the layers should be added as one would expect, by adding the first layer first,
    -- then the second, and so on. This means that we need to create a new function in
    -- each layer that propagates the input to the deepest layer and then propagates the
    -- results up to the top layer
    let (layer_forward_func, layer_initial_weights) = layer
    let new_layer_func = (\input nn ->
      let (current_layer, prev_nn) = nn
      let (prev_layer, _) = prev_nn
      let (_, current_layer_weights) = current_layer
      let prev_result = prev_layer_func input prev_layer_weights
      in layer_forward_func prev_result layer_weights
    )
    -- insert the current layer in the neural network
    let new_nn = ((new_layer_func, layer_initial_weights), nn)
    in new_nn

  let forward 'fi (nn: nn_type) (input: fi) =
    let (top_layer, nn_rest) = nn
    let (layer_function, _) = top_layer
    in layer_function input nn

  -- Loss functions, should be used for training
  -- TODO: add them
}
