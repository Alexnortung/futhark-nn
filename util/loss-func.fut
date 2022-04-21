def l1_loss [n] 't (use_sum: bool) (input: [n]t) (output[n]t) : t =
  let flat_input = flatten input
  let flat_output = flatten output
  let diffs = map2 (\x y ->
    abs (x - y)
  ) flat_input flat_output
  let sum = reduce (+) 0 diffs
  in if use_sum then
    sum
  else
    sum / n


-- when use_sum is true, it will only return the sum of the squares
-- when it is false it will return the mean
def mse_loss [n] 't (use_sum: bool) (input: [n]t) (output: [n]t) : t =
  let diffs = map2 (\x y ->
    let diff = x - y
    in diff * diff
  ) input output
  let sum = reduce (+) 0 diffs
  in if use_sum then
    sum
  else
    sum / n
