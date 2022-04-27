import "../neural-network/neural-network-new"

module nn = neural_network f64

-- ==
-- entry: sgd_test
-- input {
--    [[1.0,2.0,3.0,4.0]]
--    [[1.0,4.0,6.0,8.0]]
-- }

entry sgd_test [n] (input: [n]f64) (output: [n]f64) =
  let network = nn.init_1d n 1
  |> nn.linear (n + 2)
  |> nn.linear (n - 1)
  let loss = nn.loss.l1_loss false
  let optimizer = sgd.init 0.0001 loss network
  let optimized = sgd.train input output 100000 optimizer
  let network = optimizer.network
  in nn.forward input network
