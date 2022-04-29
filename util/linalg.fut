import "../lib/github.com/diku-dk/linalg/linalg"

module mk_linalg (T: field) = {
  open mk_linalg T

  def vecsub [n] (xs: [n]T.t) (ys: [n]T.t): *[n]T.t =
    map2 (T.-) xs ys

  def matsub [m][n] (xss: [m][n]T.t) (yss: [m][n]T.t): *[m][n]T.t =
    map2 (vecsub) xss yss

  def matmul_scalar [m][n] (xss: [m][n]T.t) (k: T.t): *[m][n]T.t =
    map (map (\x -> x T.* k)) xss

  def vecmul_scalar [n] (xss: [n]T.t) (k: T.t): [n]T.t =
    map (\x -> x T.* k) xss
}
