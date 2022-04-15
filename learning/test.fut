let f (x: f64) =
  x * x + 2

let main =
  vjp2 f 4 1
