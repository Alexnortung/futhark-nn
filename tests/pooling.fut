import "../layers/pooling"

module mypool = pooling f64

-- ==
-- entry: max_pooling
-- input { [[1.0,  2.0,  3.0,  4.0],
--          [5.0,  6.0,  7.0,  8.0],
--          [5.0,  6.0,  7.0,  8.0],
--          [9.0, 10.0, 11.0, 12.0]]}
--
-- output {[[ 6.0, 8.0 ],
--          [ 10.0, 12.0 ]]}

entry max_pooling (input: [][]f64) : [][]f64 =
  mypool.max_pool input 2 2
