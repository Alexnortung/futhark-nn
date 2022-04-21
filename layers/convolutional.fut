import "../lib/github.com/diku-dk/linalg/linalg"
import "../util/weight-initialization"
import "types"

module convolutional (R:real) = {
  type t = R.t

  type options 'stride = {
    stride: stride
  }
  type options_2d = options (i64, i64)
  type batch_2d [k] [m] [n] = [k][m][n]t
  type kernel_2d [a] [b] = [a][b]t
  type weight_bias_2d [a] [b] = (kernel_2d [a] [b], t)
  -- TODO: this works, but should be cleaner
  type^ layer_type_2d [k] [m] [n] [a] [b] [out_m] [out_n] = layer_type (options_2d) (batch_2d [k] [m] [n]) (weight_bias_2d [a] [b]) (batch_2d [k] [out_m] [out_n])

  module lalg   = mk_linalg R

  module wi = weight_init R

  let default_options_2d : options_2d = {
    stride = (1, 1)
  }

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

  let generate_weights_2d (seed: i32) (a: i64) (b: i64) : weight_bias_2d [a] [b] =
    let weights = wi.gen_2d a b seed
    let bias_max = a + b
    let bias_max : t = R.(i64 bias_max)
    let bias = wi.gen_num (R.(neg bias_max), bias_max) seed
    in (weights, bias)

  let layer_new_forward_2d [k] [m] [n] [a] [b] (output_m: i64) (output_n: i64) (_options) (input: batch_2d [k] [m] [n]) (wb: weight_bias_2d [a] [b]) : [k][output_m][output_n]t =
    let (kernel, bias) = wb
    in forward_2d output_m output_n input bias kernel

  let init_2d [k] [m] [n] (output_m: i64) (output_n: i64) (kernel_x: i64) (kernel_y: i64) (seed: i32) : layer_type_2d [k] [m] [n] [kernel_x] [kernel_y] [output_m] [output_n] =
    let forward = layer_new_forward_2d output_m output_n
    let options = default_options_2d
    let weights = generate_weights_2d seed kernel_x kernel_y
    in { forward, options, weights }

  let set_bias -- [k] [m] [n] [a] [b] [out_m] [out_n]
    -- (layer: layer_type_2d [k] [m] [n] [a] [b] )
    (bias)
    (layer)
    -- : layer_type_2d [k] [m] [n] [a] [b] =
    =
      let { forward, options, weights = (weights, _)} = layer
      in { forward, options, weights = (weights, bias)}

  let set_weights -- [k] [m] [n] [a] [b]
    -- (layer: layer_type_2d [k] [m] [n] [a] [b])
    (weights)
    (layer)
    -- : layer_type_2d [k] [m] [n] [a] [b] =
    =
      let { forward, options, weights = (_, bias)} = layer
      in { forward, options, weights = (weights, bias)}

  let forward_layer (input) (layer) =
    let { forward, options, weights } = layer
    in forward options input weights
}
