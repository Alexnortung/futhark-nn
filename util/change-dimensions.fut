module change_dimensions = {

  let from_2d_to_1d [k] [m] [n] [mn] 't (input: [k][m][n]t) : [k][mn]t =
    map (\b ->
      map (\i ->
        let y = i / m
        let x = i - y * m
        in input[b, y, x]
      ) (iota mn)
    ) (iota k)

  -- let from_1d_to_2d [k] [m] 't (input: [k][m]t) (shape: (i64, i64)) =
  --   let xdim = fst shape
  --   let ydim = snd shape
  --   let xs = iota xdim
  --   let ys = iota ydim
  --   in map (\b ->
  --     map (\y ->
  --       map (\x ->
  --         let index = y*xdim + x
  --         -- if index < m then
  --         --   input[index]
  --         -- else
  --         --   0
  --         in input[index]
  --       ) xs
  --     ) ys
  --   ) (iota k)
}
