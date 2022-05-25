module activation_func (R: real) = {
  type t = R.t

  def identity 't (n: i64) (input: [n]t) : [n]t = input

  def relu (n: i64) (input: [n]t) : [n]t =
    map (\x -> R.(max x (i32 0))) input

  def softmax (n: i64) (input: [n]t) : [n]t =
    let exp_array = map (\x -> R.(exp x)) input
    let summed = R.sum exp_array
    in map (R.((/summed))) exp_array

  def log_softmax (n: i64) (input: [n]t) : [n]t =
    softmax n input
    |> map (R.log)
}
