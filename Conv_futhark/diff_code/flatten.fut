def main [o][m][n] (Arr : [o][m][n]f32) =
    map (\outer -> flatten outer) Arr