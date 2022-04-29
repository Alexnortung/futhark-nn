import "../neural-network/neural-network-new"

module nn = neural_network f64

entry linear_optimize_test [k][n] (input: [k][n]f64) (output: [k][n]f64) =
  let net = nn.init_1d 4 1
  |> nn.linear 6 (nn.activation.identity)
  |> nn.linear 5 (nn.activation.identity)
  |> nn.linear 4 (nn.activation.identity)
  let loss = nn.make_loss nn.loss.mse net
  let optimizer = nn.optim.sgd.init 0.0001 loss net
  let optimized_net =  nn.train input output 10000 optimizer net
  in nn.forward input optimized_net
