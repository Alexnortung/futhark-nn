import "../lib/github.com/diku-dk/linalg/linalg"
import "../util/weight-initialization"
import "types"

module convolutional (R:real) = {
  type t = R.t

  type options = ()
  type batch_2d [k] [m] [n] = [k][m][n]t
  type kernel_2d [a] [b] = [a][b]t
  type weight_bias_2d [a] [b] = (kernel_2d [a] [b], t)
  -- TODO: this works, but should be cleaner
  type^ layer_type_2d [k] [m] [n] [a] [b] = (options -> batch_2d [k] [m] [n] -> weight_bias_2d [a] [b] -> ?[out_m][out_n].[k][out_m][out_n]t , options, weight_bias_2d [a] [b])
  -- type^ layer_type_2d [k] [m] [n] [a] [b] '~out = layer_type (options) (batch_2d [k] [m] [n]) (weight_bias_2d [a] [b]) (out)

  module lalg   = mk_linalg R

  module wi = weight_init R
  
  let forward_2d [k] [m] [n] [a] [b] -- k batches, m times n input nodes and a times b filter size
    (input: batch_2d [k] [m] [n])
    (bias: t)
    (filter_weights: kernel_2d [a] [b]) : [k][][]t =
      let xs = 0...(m - a)
      let ys = 0...(n - b)
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

  let forward_2d_known [k] [m] [n] [a] [b] -- k batches, m times n input nodes and a times b filter size
    (output_m: i64)
    (output_n: i64)
    (input: batch_2d [k] [m] [n])
    (bias: t)
    (filter_weights: kernel_2d [a] [b]) : [k][output_m][output_n]t =
      let xs = iota output_m
      let ys = iota output_n
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

  let generate_weights_2d (seed: i32) (a: i64) (b: i64) : ([a][b]t, t) =
    let weights = wi.gen_2d a b seed
    let bias_max = a + b
    let bias_max : t = R.(i64 bias_max)
    let bias = wi.gen_num (R.(neg bias_max), bias_max) seed
    in (weights, bias)

  let new_forward [k] [m] [n] [a] [b] _ (input: batch_2d [k] [m] [n]) (wb: weight_bias_2d [a] [b]) : [k][][]t =
    let (kernel, bias) = wb
    in forward_2d input bias kernel

  let init_2d [k] [m] [n] (kernel_x: i64) (kernel_y: i64) (seed: i32) : layer_type_2d [k] [m] [n] [kernel_x] [kernel_y] =
    (new_forward, (), generate_weights_2d seed kernel_x kernel_y)

  let set_bias [k] [m] [n] [a] [b]
    (layer: layer_type_2d [k] [m] [n] [a] [b])
    (bias: t)
    : layer_type_2d [k] [m] [n] [a] [b] =
      let (f, options, (weights, _)) = layer
      in (f, options, (weights, bias))

  let set_weights [k] [m] [n] [a] [b]
    (layer: layer_type_2d [k] [m] [n] [a] [b])
    (weights: kernel_2d [a] [b])
    : layer_type_2d [k] [m] [n] [a] [b] =
      let (f, options, (_, bias)) = layer
      in (f, options, (weights, bias))

  let forward_layer (layer) (input) =
    let (f, options, wb) = layer
    in f options input wb
}
