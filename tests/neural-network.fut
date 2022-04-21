import "../neural-network/neural-network-new"
module nn = neural_network f64

-- ==
-- entry: nn_test
-- input {
--  [[[1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0],
--    [1.0,2.0,3.0,4.0,5.0]]]
-- }
-- output {
--    [[[18.974539242722912f64, 27.177478337886157f64, 35.38041743304941f64],
--      [18.974539242722912f64, 27.177478337886157f64, 35.38041743304941f64]]]
-- }

entry nn_test [k] (input: [k][6][5]f64) =
  let n = nn.init_2d 6 5 1
  let shape = nn.conv_2d_shape 3 2 n
  let n = nn.conv_2d shape.0 shape.1 3 2 n
  let shape = nn.conv_2d_shape 3 2 n
  let n = nn.conv_2d shape.0 shape.1 3 2 n
  in nn.forward input n
