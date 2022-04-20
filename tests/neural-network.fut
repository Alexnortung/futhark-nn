import "../neural-network/neural-network-new"
module nn = neural_network f64

entry nn_test (input) =
  nn.init_2d 6 5 1
  |> nn.conv_2d 3 2
  |> nn.forward input
