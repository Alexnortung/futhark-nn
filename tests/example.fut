-- ==
-- entry: b_entry
-- input {[
--    [0.0, 0.0, 0.0, 0.0],
--    [0.0, 0.0, 0.0, 0.0],
--    [0.0, 0.0, 0.0, 0.0],
--    [0.0, 0.0, 0.0, 0.0]
--    ]
--
--    2i64
--    2i64
-- }
-- output {[
--    [0.0, 0.0],
--    [0.0, 0.0]
-- ]}
type ret [out_m] [out_n] 't = [out_m][out_n]t

def b [m] [n]  (output_m: i64) (output_n: i64) (input: [m][n]f64) : ret [output_m] [output_n] f64 =
  let w = m / output_m
  let h = n / output_n
  let xs = map (\x -> x * w) (iota output_n)
  let ys = map (\y -> y * h) (iota output_m)
  in map (\y ->
    map (\x ->
      input[y, x]
    ) xs
  ) ys

def c = uncurry b

entry b_entry [m] [n] (input: [m][n]f64) (output_m: i64) (output_n: i64) : ret [output_m] [output_n] f64 =
  -- let s = (output_m, output_n)
  -- let b1 = uncurry b
  -- in b1 s input
  c (output_m, output_n) input

