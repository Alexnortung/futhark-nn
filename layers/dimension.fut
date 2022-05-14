import "../util/change-dimensions"
import "types"

module dimension = {
  type options = ()

  type^ dimension_layer 'from_size 'to_size 'shape = layer_type () options from_size () shape to_size

  def options = ()
  def weights = ()
  def apply_optimize (_options) (_) (w) (_wg) = w

  def from_1d_2d [k] [n] 't (output_m: i64) (output_n: i64) : dimension_layer ([k][n]t) ([k][output_m][output_n]t) shape_2d =
    let forward (_options) (input: [k][n]t) (_weights) : ([k][output_m][output_n]t) =
      change_dimensions.from_1d_to_2d output_m output_n input
    in {
      forward, options, weights, apply_optimize,
      shape = (output_m, output_n)
    }

  def from_1d_3d [k] [n] 't (output_l: i64) (output_m: i64) (output_n: i64) : dimension_layer ([k][n]t) ([k][output_l][output_m][output_n]t) shape_3d =
    let forward (_options) (input: [k][n]t) (_weights) : ([k][output_l][output_m][output_n]t) =
      change_dimensions.from_1d_to_3d output_l output_m output_n input
    in {
      forward, options, weights, apply_optimize,
      shape = (output_l, output_m, output_n)
    }

  def from_2d_1d [k] [m] [n] 't (mn: i64) : dimension_layer ([k][m][n]t) ([k][mn]t) shape_1d =
    let forward (_options) (input: [k][m][n]t) (_weights) =
      change_dimensions.from_2d_to_1d input
    in {
      forward, options, weights, apply_optimize,
      shape = mn
    }

  def from_3d_1d [k] [l] [m] [n] 't (lmn: i64) : dimension_layer ([k][l][m][n]t) ([k][lmn]t) shape_1d =
    let forward (_options) (input: [k][l][m][n]t) (_weights) =
      change_dimensions.from_3d_to_1d input
    in {
      forward, options, weights, apply_optimize,
      shape = lmn
    }

  def forward_layer (input) (layer) =
    let { forward, apply_optimize, options, weights, shape = _ } = layer
    in forward options input weights
}

-- TESTS

local module test_dim = dimension

-- ==
-- entry: test_1d_2d
-- input { [[1.0, 2.0]] 2i64 1i64 }
-- output { [[[1.0],[2.0]]] }
entry test_1d_2d (input) (x) (y) : [][x][y]f64 =
  test_dim.from_1d_2d x y
  |> test_dim.forward_layer input

-- ==
-- entry: test_1d_3d
-- input { [[1.0, 2.0, 3.0]] 1i64 3i64 1i64 }
-- output { [[[[1.0],[2.0], [3.0]]]] }
entry test_1d_3d (input) (x) (y) (z) : [][x][y][z]f64 =
  test_dim.from_1d_3d x y z
  |> test_dim.forward_layer input
