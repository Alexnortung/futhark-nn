-- Small example of training a MLP
--
-- Run 'make' to generate data sets.
-- ==
-- compiled input @ batch_16_mnist_100000_f32.bindata
-- compiled input @ batch_32_mnist_100000_f32.bindata
-- compiled input @ batch_64_mnist_100000_f32.bindata
-- compiled input @ batch_128_mnist_100000_f32.bindata

import "../neural-network/neural-network-new"

module nn = neural_network f32
let seed = 1

entry main [K] (batch_size: i32) (input:[K][784]dl.t) (labels: [K][10]dl.t) =
  -- initialize network
  let net = nn.init_1d 784
  |> nn.add_layer (nn.dimension.from_1d_3d 1 28 28)
  |> nn.conv_2d 26 26 32 3 3 (nn.activation.relu)
  |> nn.conv_2d 24 24 64 3 3 (nn.activation.relu)
  |> nn.add_layer (nn.dimension.from_3d_2d 22 (22 * 64))
  |> nn.maxpool_2d 11 (11 * 64) -- window size 2 2
  |> nn.add_layer (nn.dimension.from_2d_1d (11 * 11 * 64))
  |> nn.linear 128 (nn.activation.relu)
  |> nn.linear 10 (nn.activation.log_softmax)
  -- train the network
