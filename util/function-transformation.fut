module function_transformation = {
  let args_to_tuple_2 't1 't2 't (function: t1 -> t2 -> t) : (t1, t2) -> t =
    (\(x1, x2) -> function x1 x2)

  let tuple_to_args_2 't1 't2 't (function: (t1, t2) -> t) : t1 -> t2 -> t =
    (\x1 x2 -> function x1 x2)
}
