import "../layers/linear"

module linear_f64 = linear f64

-- ==
-- entry: linear_forward
-- input {[
--        [1.0, 2.0, 3.0, 4.0],
--        [2.0, 3.0, 4.0, 5.0],
--        [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[
--          [ 31.0,  72.0, 113.0],
--          [ 41.0,  98.0, 155.0],
--          [ 51.0, 124.0, 197.0]
--         ]}

entry linear_forward [k] (input: [k][]f64) w b : [k][]f64 =
  let output = linear_f64.forward input (\x -> x) w b
  in output

-- ==
-- entry: linear_init_forward
-- input {[
--        [1.0, 2.0, 3.0, 4.0],
--        [2.0, 3.0, 4.0, 5.0],
--        [3.0, 4.0, 5.0, 6.0]]
--
--        [[1.0,  2.0,  3.0,  4.0],
--         [5.0,  6.0,  7.0,  8.0],
--         [9.0, 10.0, 11.0, 12.0]]
--
--         [1.0, 2.0, 3.0]}
--
-- output {[
--          [ 31.0,  72.0, 113.0],
--          [ 41.0,  98.0, 155.0],
--          [ 51.0, 124.0, 197.0]
--         ]}

entry linear_init_forward [k] [m] [n] (input: [k][m]f64) (w: [n][m]f64) (b: [n]f64) : [k][n]f64 =
  let layer = linear_f64.init m n (\x -> x) 1
  let layer = linear_f64.set_weights layer w
  let layer = linear_f64.set_bias layer b
  let output = linear_f64.forward_layer layer input
  in output
