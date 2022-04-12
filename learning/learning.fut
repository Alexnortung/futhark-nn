import "../layers/convolutional"
import "../layers/fully-connected"
import "../layers/pooling"
import "../util/change-dimensions"
import "../util/weight-initialization"
import "../util/activation-func"

module wi = weight_init f64
module myconv = convolutional f64
module mypool = pooling f64
module fcs = fully_connected_simple f64

let abs (x: f64) : f64 =
  if x < 0 then -x else x

let nn [k] [m] [n] [w1x] [w1y] [m3] [n3] (input: [k][m][n]f64) (w1: [w1x][w1y]f64, b1: f64, w3: [n3][m3]f64, b3: [n3]f64) : [k][n3]f64 =
  let out = myconv.forward input b1 w1
  let out = mypool.forward out 2 2
  let out = change_dimensions.from_2d_to_1d out
  let out = fcs.forward out (identity) w3 b3
  in out

let nn_loss [k] [m] [n] [w1x] [w1y] [m3] [n3] (input: [k][m][n]f64) (output: [k][n3]f64) (w1: [w1x][w1y]f64, b1: f64, w3: [n3][m3]f64, b3: [n3]f64) : f64 =
  let result = nn input (w1, b1, w3, b3)
  in map2 (\x o ->
    map2 (\x o ->
      let diff = abs (x - o)
      in diff * diff
    ) x o
  ) result output
  |> flatten
  |> reduce (+) 0
  |> f64.sqrt

-- let newton 'a 'b (fun: a -> b) (x_n: a) : a =
--   let jacobian = vjp2 fun
--   let (result, x_n1) = jacobian x_n

let main n =
  let input =
   [[[ 1.0, 2.0, 3.0,  1.0, 2.0],
     [10.0, 9.0, 8.0, 10.0, 9.0],
     [ 4.0, 5.0, 6.0, 4.0, 5.0],
     [ 1.0, 2.0, 3.0,  1.0, 2.0],
     [ 4.0, 5.0, 6.0, 4.0, 5.0]],
    [[ 1.0, 2.0, 6.0, 4.0, 4.0],
     [10.0, 11.0, 8.0, 10.0, 14.0],
     [ 4.0, 8.0, 10.0, 7.0, 15.0],
     [10.0, 9.0, 8.0, 10.0, 12.0],
     [ 4.0, 5.0, 7.0, 4.0, 13.0]]]
  let output = [[112], [219]]
  let seed = 1
  let w1 = wi.gen_2d 2 2 seed
  let b1 = wi.gen_num (-1000, 1000) seed
  let w3 = wi.gen_2d 2 1 seed
  let b3 = wi.gen_1d 1 seed
  let x_0 = (w1, b1, w3, b3)
  let alpha = 0.001
  -- let fixed_nn = (nn input)
  let fixed_nn = nn_loss input output
  -- let reversed_nn = vjp2 fixed_nn (w1, b1, w3, b3) [[ 1 ], [ 1 ]]
  -- let (result, (neww1, newb1, neww3, newb3)) = reversed_nn
  -- let n =4 
  let (result, x_n) = loop (result, x_n) = (fixed_nn x_0, x_0) for i < n do
    let (result, x_np) = vjp2 fixed_nn x_n 1
    let x_n1 = (
      map2 (\x p ->
        map2 (\x p -> x - alpha * p) x p
      ) x_n.0 x_np.0,
      x_n.1 - alpha * x_np.1,
      map2 (\x p ->
        map2 (\x p -> x - alpha * p) x p
      ) x_n.2 x_np.2,
      map2 (\x p -> x - alpha * p) x_n.3 x_np.3
    )
    in (result, x_n1)
  -- in fixed_nn (neww1, newb1, neww3, newb3)
  -- in fixed_nn (w1, b1, w3, b3)
  -- let x_1 = x_0 - x_0p
  -- let result1 = fixed_nn x_1
  in nn input x_n
