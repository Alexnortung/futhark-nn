module change_dimensions = {
  def from_1d_to_2d [k] [m] 't (output_m: i64) (output_n: i64) (input: [k][m]t) =
    let ms = iota output_m
    let ns = iota output_n
    in map (\b ->
      map (\m ->
        map (\n ->
          let index = m * output_n + n
          in input[b, index]
        ) ns
      ) ms
    ) (iota k)

  def from_1d_to_3d [k] [m] 't (output_l: i64) (output_m: i64) (output_n: i64) (input: [k][m]t) =
    let ls = iota output_l
    let ms = iota output_m
    let ns = iota output_n
    in map (\b ->
      map (\l ->
        map (\m ->
          map (\n ->
            let index = l * output_m * output_n + m * output_n + n
            in input[b, index]
          ) ns
        ) ms
      ) ls
    ) (iota k)

  -- TODO: make sure that mn is always inferred correctly
  def from_2d_to_1d [k] [m] [n] 't (mn: i64) (input: [k][m][n]t) : [k][mn]t =
    map (\b ->
      map (\i ->
        let y = i / n
        let x = i - y * n
        in input[b, y, x]
      ) (iota mn)
    ) (iota k)

  def from_3d_to_1d [k] [l] [m] [n] 't (lmn: i64) (input: [k][l][m][n]t) : [k][lmn]t =
    map (\b ->
      map (\i ->
        let mn = m * n
        let z = i / mn
        let i2d = i - z * mn
        let y = i2d / n
        let x = i2d - y * n
        in input[b, z, y, x]
      ) (iota lmn)
    ) (iota k)

  def from_3d_to_2d [k] [l] [m] [n] 't (out_m: i64) (out_n: i64) (input: [k][l][m][n]t) : [k][out_m][out_n]t =
    from_3d_to_1d (l * m * n) input
    |> from_1d_to_2d out_m out_n

}
