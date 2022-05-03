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
  type options_1d = options_type
  type options_2d = options_type
  type options_3d = options_type
  type^ layer_type_1d [k] [n] [out_n] = layer_type t options_1d (batch_1d [k] [n]) () i64 ([k][out_n]t)
  type^ layer_type_2d [k] [m] [n] [out_m] [out_n] = layer_type t options_2d (batch_2d [k] [m] [n]) () (i64, i64) ([k][out_m][out_n]t)

  def arg_max_1d [n] (input: input_1d [n]) : t =
    R.(reduce (\acc x ->
      if acc > x
      then acc
      else x) input[0] input)

  def arg_max_2d [m] [n] (input: input_2d [m] [n]) : t =
    -- flatten the slice to one dimensional
    let flat = flatten input
    -- compare and find greatest element
    in arg_max_1d flat

  def arg_max_3d [l] [m] [n] (input: input_3d [l] [m] [n]) : t =
    let flat = flatten input
    in arg_max_2d flat

  def forward_1d [k] [n]
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

  def forward_2d [k] [m] [n]
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

  def layer_forward_1d [k] [n]
    (output_n: i64)
    (_options: options_1d)
    (input: batch_1d [k] [n])
    (_weights)
    : batch_1d [k] [output_n] =
      forward_1d output_n input

  def layer_forward_2d [k] [m] [n]
    (output_m: i64)
    (output_n: i64)
    (_options: options_2d)
    (input: batch_2d [k] [m] [n])
    (_weights)
    : batch_2d [k] [output_m] [output_n] =
      forward_2d output_m output_n input

  def apply_optimize
    (_options)
    (_apply_func_record: optimizer_apply_record t)
    (current_weights)
    (_gradient_weights) =
      current_weights


  def init_1d [k] [n] (output_n: i64) : layer_type_1d [k] [n] [output_n] =
    { forward = layer_forward_1d output_n,
      apply_optimize,
      options = (), weights = (),
      shape = output_n }

  def init_2d [k] [m] [n] (output_m: i64) (output_n: i64) : layer_type_2d [k] [m] [n] [output_m] [output_n] =
    { forward = layer_forward_2d output_m output_n,
      apply_optimize,
      options = (), weights = (),
      shape = (output_m, output_n) }

  def forward_layer (input) (layer) =
    let { forward, options, apply_optimize, weights, shape = _ } = layer
    in forward options input weights

}

-- TESTS

local module test_pool = pooling f64

-- ==
-- entry: max_pooling_1d max_pooling_1d_layer
-- input {
--    [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
-- }
-- output {
--    [[3.0, 6.0]]
-- }
entry max_pooling_1d [k] (input: [k][]f64) : [k][]f64 =
  test_pool.forward_1d 2 input

entry max_pooling_1d_layer [k] [n] (input: [k][n]f64) : [k][]f64 =
  test_pool.init_1d 2
  |> test_pool.forward_layer input

-- ==
-- entry: max_pooling_2d max_pooling_2d_layer
-- input { [
--         [[1.0,  2.0,  3.0,  4.0],
--          [5.0, 26.0, 27.0,  8.0],
--          [5.0, 16.0,  7.0,  8.0],
--          [9.0, 10.0, 91.0, 12.0]],
--         [[1.0,  2.0,  3.0,  4.0],
--          [5.0,  6.0,  7.0,  8.0],
--          [5.0,  6.0,  7.0,  8.0],
--          [9.0, 10.0, 11.0, 12.0]]] }
--
-- output {[
--         [[ 26.0, 27.0 ],
--          [ 16.0, 91.0 ]],
--         [[ 6.0, 8.0 ],
--          [ 10.0, 12.0 ]]
--         ]}
entry max_pooling_2d [k] (input: [k][][]f64) : [k][][]f64 =
  test_pool.forward_2d 2 2 input

entry max_pooling_2d_layer [k] [m] [n] (input: [k][m][n]f64) : [k][][]f64 =
  test_pool.init_2d 2 2
  |> test_pool.forward_layer input

