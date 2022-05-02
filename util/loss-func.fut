module loss (T: numeric) = {
  type t = T.t

  def l1 (use_sum: bool) (n: i64) (input: [n]t) (output: [n]t) : t =
    let diffs = map2 (\x y ->
      T.(abs (x - y))
    ) input output
    let summed = T.sum diffs
    in if use_sum then
      summed
    else
      T.(summed / (i64 n))


  -- when use_sum is true, it will only return the sum of the squares
  -- when it is false it will return the mean
  def mse (use_sum: bool) (n: i64) (input: [n]t) (output: [n]t) : t =
    let diffs = map2 (\x y ->
      let diff = T.(x - y)
      in T.(diff * diff)
    ) input output
    let summed = T.sum diffs
    in
    if use_sum then summed
    else T.(summed / i64 n)
}
