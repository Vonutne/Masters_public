
def main [m][n][k] (A: [m][n]f32 ) (B: [n][k]f32)  =
    map (\Arow -> map (\Bcol -> map2 (*) Arow Bcol |> reduce (+) 0.0 ) (transpose B)) A
