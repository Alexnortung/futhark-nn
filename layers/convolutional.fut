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
      let flat_weights = (flatten filter_weights) :> [c]t
      in
      map (\input ->
        map (\x -> 
          map (\y ->
            let input_slice = input[x:x+a,y:y+b]
            let flat_slice = (flatten input_slice) :> [c]t
            let dotted = lalg.dotprod flat_slice flat_weights
            in dotted
          ) ys
        ) xs
      ) input
}
