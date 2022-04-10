import "../util/change-dimensions"
import "../util/weight-initialization"

module wi = weight_init f64

let nn [k] [m] [n] [w1x] [w1y] [m3] [n3] (input: [k][m][n]f64) (w1: [w1x][w1y]f64) (b1: f64) (w3: [n3][m3]f64) (b3: [n3]f64) : [k][n3]f64 =
  let out = myconv.forward input b1 w1
  let out = mypool.forward out 2 2
  let out = change_dimensions.from_2d_to_1d out
  let out = fcs.forward out (identity) w3 b3
  in out

let main () =
  let seed = 1
  let w1 = wi.gen_2d 2 2 seed
  let b1 = wi.gen_num (-1000, 1000) seed
  let w3 = wi.gen_2d 2 1 seed
  let b3 = wi.gen_1d (-1000, 1000) 1 seed
  let reversed_nn = vjp nn (w1, b1, w3, b3) 1
  in reversed_nn
