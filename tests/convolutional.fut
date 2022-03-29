import "../layers/convolutional"

module myconv = convolutional f64

-- ==
-- entry: conv
-- input {[
--        [[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 6.0]],
--        [[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 7.0]]
--        ]
--      
--        [[1.0, 3.0],
--         [2.0, 9.0]]
-- }
-- output {[
--         [[108.0, 101.0],
--          [ 90.0,  97.0]],
--         [[108.0, 101.0],
--          [ 90.0, 106.0]]
--         ]
-- }

entry conv [k] (input: [k][][]f64) (weights: [][]f64) : [k][][]f64 =
  myconv.forward input weights