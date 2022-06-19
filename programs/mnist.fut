import "../neural-network/neural-network"

module nn = neural_network f64
let seed = 1i32

def format_labels (labels) =
  map (\l -> 
    let inner = replicate 10 0.0f64
    let i = f64.to_i64 l
    in scatter inner [i] [1.0f64]
  ) labels

entry main [k] [k_test] (input: [k][1][28][28]f64) (labels: [k]f64) (test_input: [k_test][1][28][28]f64) (test_labels: [k_test]f64)  =
  -- transform labels
  let labels = format_labels labels
  let test_labels = format_labels test_labels
  -- initialize network
  let net = nn.init_3d 1 28 28 seed
  |> nn.conv_2d 26 26 32 3 3 (nn.activation.relu)
  |> nn.conv_2d 24 24 64 3 3 (nn.activation.relu)
  |> nn.add_layer (nn.layers.dimension.from_3d_2d 24 (24 * 64))
  |> nn.maxpool_2d 12 (12 * 64) -- window size 2 2
  |> nn.add_layer (nn.layers.dimension.from_2d_1d (12 * 12 * 64))
  |> nn.linear 128 (nn.activation.relu)
  |> nn.linear 10 (nn.activation.log_softmax)
  -- train the network
  let loss = nn.make_loss (nn.loss.mse false) net
  let optimizer = nn.optim.sgd.init 0.01 (loss)
  let net = nn.train input labels 50 optimizer net
  in nn.accuracy test_input test_labels (nn.activation.identity) net
  
