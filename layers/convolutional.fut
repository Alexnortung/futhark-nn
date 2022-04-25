import "../lib/github.com/diku-dk/linalg/linalg"
import "../util/weight-initialization"
import "types"

module convolutional (R:real) = {
  type t = R.t

  type options 'stride = {
    stride: stride
  }
  type options_1d = options (shape_1d)
  type options_2d = options (shape_2d)
  type batch_1d [k] [n] = [k][n]t
  type batch_2d [k] [m] [n] = [k][m][n]t
  type kernel_1d [a] = [a]t
  type kernel_2d [a] [b] = [a][b]t
  type weight_bias_1d [a] = (kernel_1d [a], t)
  type weight_bias_2d [a] [b] = (kernel_2d [a] [b], t)
  -- TODO: this works, but should be cleaner
  type^ layer_type_1d [k] [n] [a] [out_n] = layer_type (options_1d) (batch_1d [k] [n]) (weight_bias_1d [a]) (shape_1d) (batch_1d [k] [out_n])
  type^ layer_type_2d [k] [m] [n] [a] [b] [out_m] [out_n] = layer_type (options_2d) (batch_2d [k] [m] [n]) (weight_bias_2d [a] [b]) (i64, i64) (batch_2d [k] [out_m] [out_n])

  module lalg   = mk_linalg R

  module wi = weight_init R

  let default_options_1d : options_1d = {
    stride = 1
  }
  let default_options_2d : options_2d = {
    stride = (1, 1)
  }

  let forward_1d [k] [n] [a]
    (output_n: i64)
    (input: batch_1d [k] [n])
    (bias: t)
    (filter_weights: kernel_1d [a]) : batch_1d [k] [output_n] =
      let xs = iota output_n
      in
      map (\input ->
        map (\x ->
          let slice = input[x:x+a] :> [a]t
          let dotted = lalg.dotprod slice filter_weights
          in R.(dotted + bias)
        ) xs
      ) input

  let forward_2d [k] [m] [n] [a] [b] -- k batches, m times n input nodes and a times b filter size
    (output_m: i64)
    (output_n: i64)
    (input: batch_2d [k] [m] [n])
    (bias: t)
    (filter_weights: kernel_2d [a] [b]) : batch_2d [k] [output_m] [output_n] =
      let xs = iota output_m -- m - a + 1
      let ys = iota output_n -- n - b + 1
      let c = a * b
      -- TODO: find a solution where there is not used dynamic casting
      let flat_weights = (flatten filter_weights) :> [c]t
      in
      map (\input ->
        map (\x -> 
          map (\y ->
            let input_slice = input[x:x+a,y:y+b]
            let flat_slice = (flatten input_slice) :> [c]t
            let dotted : t = lalg.dotprod flat_slice flat_weights
            in R.(dotted + bias)
          ) ys
        ) xs
      ) input

  let generate_weights_1d (seed: i32) (a: i64) : weight_bias_1d [a] =
    let weights = wi.gen_1d a seed
    let bias_max : t = R.(i64 a)
    let bias = wi.gen_num (R.(neg bias_max), bias_max) seed
    in (weights, bias)

  let generate_weights_2d (seed: i32) (a: i64) (b: i64) : weight_bias_2d [a] [b] =
    let weights = wi.gen_2d a b seed
    let bias_max = a + b
    let bias_max : t = R.(i64 bias_max)
    let bias = wi.gen_num (R.(neg bias_max), bias_max) seed
    in (weights, bias)

  let layer_new_forward_1d [k] [n] [a] (output_n: i64) (_options) (input: batch_1d [k] [n]) (wb: weight_bias_1d [a]) : batch_1d [k] [output_n] =
    let (kernel, bias) = wb
    in forward_1d output_n input bias kernel

  let layer_new_forward_2d [k] [m] [n] [a] [b] (output_m: i64) (output_n: i64) (_options) (input: batch_2d [k] [m] [n]) (wb: weight_bias_2d [a] [b]) : [k][output_m][output_n]t =
    let (kernel, bias) = wb
    in forward_2d output_m output_n input bias kernel

  let init_1d [k] [n] (output_n: i64) (kernel_n: i64) (seed: i32) : layer_type_1d [k] [n] [kernel_n] [output_n] =
    let forward = layer_new_forward_1d output_n
    let options = default_options_1d
    let weights = generate_weights_1d seed kernel_n
    in { forward, options, weights, shape = output_n }

  let init_2d [k] [m] [n] (output_m: i64) (output_n: i64) (kernel_x: i64) (kernel_y: i64) (seed: i32) : layer_type_2d [k] [m] [n] [kernel_x] [kernel_y] [output_m] [output_n] =
    let forward = layer_new_forward_2d output_m output_n
    let options = default_options_2d
    let weights = generate_weights_2d seed kernel_x kernel_y
    in { forward, options, weights, shape = (output_m, output_n) }

  let set_bias -- [k] [m] [n] [a] [b] [out_m] [out_n]
    -- (layer: layer_type_2d [k] [m] [n] [a] [b] )
    (bias)
    (layer)
    -- : layer_type_2d [k] [m] [n] [a] [b] =
    =
      let { forward, options, shape, weights = (weights, _)} = layer
      in { forward, options, shape, weights = (weights, bias)}

  let set_weights -- [k] [m] [n] [a] [b]
    -- (layer: layer_type_2d [k] [m] [n] [a] [b])
    (weights)
    (layer)
    -- : layer_type_2d [k] [m] [n] [a] [b] =
    =
      let { forward, options, shape, weights = (_, bias)} = layer
      in { forward, options, shape, weights = (weights, bias)}

  let forward_layer (input) (layer) =
    let { forward, options, weights, shape = _ } = layer
    in forward options input weights
}
