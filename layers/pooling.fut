module pooling (R:real) = {
  --type t = R.t
  type t = f64
  let arg_max [m] [n] (input: [m][n]t) : t =
    -- flatten the slice to one dimensional
    let flat = flatten input
    -- compare and find greatest element
    in reduce (\acc x ->
      if acc > x
      then acc
      else x) flat[0] flat

  let max_pool [k] [m] [n]
    (input: [k][m][n]t)
    (output_m: i64)
    (output_n: i64) : [k][output_m][output_n]t =
      map (\input ->
        -- find the window width and height from the given output sizes
        let window_width = m / output_m
        let window_height = n / output_n
        -- find all the indexes in each dimension
        let xs = map (\x -> x * window_width) (0..<output_m)
        let ys = map (\y -> y * window_height) (0..<output_n)
        -- calculate the 
        in map (\x ->
          map (\y -> 
            let slice = input[x:x+window_width, y:y+window_height]
            in arg_max slice
          ) ys
        ) xs
      ) input
}
