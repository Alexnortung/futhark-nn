import "../neural-network/neural-network"

module nn = neural_network f64

-- ==
-- entry: linear_optimize_test
-- input {
--    [[1.0, 2.0, 3.0, 4.0]]
--    [[2.0, 4.0, 6.0, 8.0]]
--    10000i64
-- }
-- output {
--    [[2.0, 4.0, 6.0, 8.0]]
-- }
entry linear_optimize_test [k] (input: [k][4]f64) (output: [k][4]f64) (iterations: i64) =
  let net = nn.init_1d 4i64 1i32
  |> nn.linear 6 (nn.activation.identity)
  |> nn.linear 5 (nn.activation.identity)
  |> nn.linear 4 (nn.activation.identity)
  let loss = nn.make_loss (nn.loss.mse false) net
  let optimizer = nn.optim.sgd.init 0.0001 loss
  let optimized_net =  nn.train input output iterations optimizer net
  in nn.forward input optimized_net
  -- in loss input output net.weights
  -- in nn.forward input net
