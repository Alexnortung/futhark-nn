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
--         [ 4.0, 5.0, 7.0]]]
--
--        3.0
--      
--        [[1.0, 3.0],
--         [2.0, 9.0]]
-- }
-- output {[
--         [[111.0, 104.0],
--          [ 93.0, 100.0]],
--         [[111.0, 104.0],
--          [ 93.0, 109.0]]
--         ]
-- }

entry conv [k] (input: [k][][]f64) (bias: f64) (weights: [][]f64) : [k][][]f64 =
  myconv.forward_2d input bias weights


-- ==
-- entry: conv_layer
-- input {[
--        [[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 6.0]],
--        [[ 1.0, 2.0, 3.0],
--         [10.0, 9.0, 8.0],
--         [ 4.0, 5.0, 7.0]]]
--
--        3.0
--      
--        [[1.0, 3.0],
--         [2.0, 9.0]]
-- }
-- output {[
--         [[111.0, 104.0],
--          [ 93.0, 100.0]],
--         [[111.0, 104.0],
--          [ 93.0, 109.0]]
--         ]
-- }

entry conv_layer [k] [a] [b] (input: [k][][]f64) (bias: f64) (weights: [a][b]f64) : [k][][]f64 =
  let layer = myconv.init_2d a b 1
  let layer = myconv.set_bias layer bias
  let layer = myconv.set_weights layer weights
  in myconv.forward_layer layer input
