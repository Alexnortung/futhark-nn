import "../lib/github.com/diku-dk/linalg/linalg"

module convolutional (R:real) = {
  type t = R.t

  module lalg   = mk_linalg R
  
  let forward [k] [m] [n] [a] [b] -- k batches, m times n input nodes and a times b filter size
    (input: [k][m][n]t)
    (filter_weights: [a][b]t) : [k][][]t =
      let xs = 0...(m-a)
      let ys = 0...(n-b)
      let c = a * b
      -- TODO: make this work properly
      let flattened_weights = flatten filter_weights
      let flat_weights = iota c
      let flat_weights: [c]t = map (\i -> flattened_weights[i]) flat_weights
      in
      map (\input ->
        map (\x -> 
          map (\y ->
            let input_slice = input[x:x+a,y:y+b]
            let flattened_slice = flatten input_slice
            let flat_slice = iota c
            let flat_slice: [c]t = map (\i -> flattened_slice[i]) flat_slice
            let dotted = lalg.dotprod flat_slice flat_weights
            in dotted
          ) ys
        ) xs
      ) input
}
