import "../neural-network/neural-network-new"

module nn = neural_network f64

entry linear_optimize_test [k] (input: [k][4]f64) (output: [k][4]f64) =
  let net = nn.init_1d 4i64 1i32
  |> nn.linear 6 (nn.activation.identity)
  |> nn.linear 5 (nn.activation.identity)
  |> nn.linear 4 (nn.activation.identity)
  let loss = nn.make_loss (nn.loss.mse false) net
  let optimizer = nn.optim.sgd.init 0.0001 loss
  let optimized_net =  nn.train input output 10000 optimizer net
  in nn.forward input optimized_net
  -- in loss input output net.weights
  -- in nn.forward input net
