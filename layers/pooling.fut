import "types"

module pooling (R:real) = {
  type t = R.t
  type input_1d [n] = [n]t
  type input_2d [m] [n] = [m][n]t
  type input_3d [l] [m] [n] = [l][m][n]t
  type batch_1d [k] [n] = [k](input_1d [n])
  type batch_2d [k] [m] [n] = [k](input_2d [m] [n])
  type batch_3d [k] [l] [m] [n] = [k](input_3d [l] [m] [n])

  type options_type = ()
  type options_2d = options_type
  type^ layer_type_2d [k] [m] [n] [out_m] [out_n] = layer_type options_2d (batch_2d [k] [m] [n]) () ([k][out_m][out_n]t)

  -- type t = f64
  let arg_max_1d [n] (input: input_1d [n]) : t =
    R.(reduce (\acc x ->
      if acc > x
      then acc
      else x) input[0] input)

  let arg_max_2d [m] [n] (input: input_2d [m] [n]) : t =
    -- flatten the slice to one dimensional
    let flat = flatten input
    -- compare and find greatest element
    in arg_max_1d flat

  let arg_max_3d [l] [m] [n] (input: input_3d [l] [m] [n]) : t =
    let flat = flatten input
    in arg_max_2d flat

  let forward_1d [k] [n]
    (output_n: i64)
    (input: batch_1d [k] [n])
    : [k][output_n]t =
      let window_width = n / output_n
      let xs = map (*window_width) (iota output_n)
      in map (\input ->
        map (\x ->
          let slice = input[x:x+window_width]
          in arg_max_1d slice
        ) xs
      ) input

  let forward_2d [k] [m] [n]
    (output_m: i64)
    (output_n: i64)
    (input: batch_2d [k] [m] [n])
    : [k][output_m][output_n]t =
      -- find the window width and height from the given output sizes
      let window_width = m / output_m
      let window_height = n / output_n
      -- find all the indexes in each dimension
      let xs = map (\x -> x * window_width) (0..<output_m)
      let ys = map (\y -> y * window_height) (0..<output_n)
      in map (\input ->
        map (\x ->
          map (\y -> 
            let slice = input[x:x+window_width, y:y+window_height]
            in arg_max_2d slice
          ) ys
        ) xs
      ) input

  let layer_new_forward_2d [k] [m] [n]
    (output_m: i64)
    (output_n: i64)
    (_options: options_2d)
    (input: batch_2d [k] [m] [n])
    (_weights)
    : batch_2d [k] [output_m] [output_n] =
      forward_2d output_m output_n input

  let init_2d [k] [m] [n] (output_m: i64) (output_n: i64) : layer_type_2d [k] [m] [n] [output_m] [output_n] =
    -- let output_m = m / window_width
    -- let output_n = n / window_height
    { forward = layer_new_forward_2d output_m output_n, options = (), weights = () }

  let forward_layer (input) (layer) =
    let { forward, options, weights } = layer
    in forward options input weights

}
