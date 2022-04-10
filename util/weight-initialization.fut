import "../lib/github.com/diku-dk/cpprandom/random"

module weight_init (R: real) = {
  type t = R.t

  module rand = uniform_real_distribution R minstd_rand

  let gen_num (seed: i32) : t =
    let rng = minstd_rand.rng_from_seed [seed]
    let (_, x) = rand.rand dist rng
    in x

  -- let gen_num (seed: i32) : t =
  --   let rng = minstd_rand.rng_from_seed [seed]
  --   let (_, x) = rand.rand (-max, max) rng
  --   in x

  let gen_1d (n: i64) (seed: i32) : [n]t =
    map (\_ -> gen_num seed) (iota n)

  let gen_2d (m: i64) (n: i64) (seed: i32) : [n][m]t =
    map (\_ ->
      gen_1d m seed
    ) (iota n)
}