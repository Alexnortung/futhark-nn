import "../layers/fully-connected"

module fcs = fully_connected_simple f64

-- ==
-- entry: fully_connected_simple_fwd
-- input {[
--        [1.0, 2.0, 3.0, 4.0],
--        [2.0, 3.0, 4.0, 5.0],
--        [3.0, 4.0, 5.0, 6.0]
--        ]
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

entry fully_connected_simple_fwd [k] (input: [k][]f64) w b : [k][]f64 =
  let output = fcs.forward input (\x -> x) w b
  in output
