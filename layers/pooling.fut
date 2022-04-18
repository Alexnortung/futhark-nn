import "types"

module pooling (R:real) = {
  type t = R.t
  type input_1d [n] = [n]t
  type input_2d [m] [n] = [m][n]t
  type input_3d [l] [m] [n] = [l][m][n]t
  type batch_1d [k] [n] = [k](input_1d [n])
  type batch_2d [k] [m] [n] = [k](input_2d [m] [n])
  type batch_3d [k] [l] [m] [n] = [k](input_3d [l] [m] [n])

  type options_type 'shape = { shape: shape } -- shape of outputs
  type options_2d = options_type (i64, i64)
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
    let flat = flatten flat
    in arg_max_1d flat

  let forward_1d [k] [n]
    (input: batch_1d [k] [n])
    (output_n: i64)
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
    (input: batch_2d [k] [m] [n])
    (output_m: i64)
    (output_n: i64) : [k][output_m][output_n]t =
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

  -- let create_2d [k] [m] [n] (width: i64) (height: i64) (input: batch_2d [k] [m] [n]) () : [k][][]t =
  --   -- same as out_n and out_m
  --   let output_m = m / width
  --   let output_n = n / height
  --   let new_forward = forward_2d input output_m output_n
  --   in new_forward
  let new_forward [k] [m] [n]
    ({ shape = (output_m, output_n) }: options_2d)
    (input: batch_2d [k] [m] [n])
    ()
    : batch_2d [k] [] [] =
      let out = forward_2d input output_m output_n
      in out

  let init_2d [k] [m] [n] (window_width: i64) (window_height: i64) : layer_type_2d [k] [m] [n] [] [] =
    -- let new_forward = create_2d window_width window_height
    let output_m = m / window_width
    let output_n = n / window_height
    let options = { shape = (output_m, output_n) }
    in (new_forward, options, ())

  -- let forward_layer [k] 'options_type 'input_type 'wb 'out (layer: layer_type options_type ([k]input_type) wb out) (input: [k]input_type) : out =
  --   let (func, options, func_inputs) = layer
  --   in func options input func_inputs

  let forward_layer_2d [k] [m] [n] (layer: layer_type_2d [k] [m] [n] [] []) (input: batch_2d [k] [m] [n]) : batch_2d [k] [] [] =
    let (_, options, _) = layer
    let (output_m, output_n) = options.shape
    in forward_2d input output_m output_n
    -- in func options input func_inputs

}
