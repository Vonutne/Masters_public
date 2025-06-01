  
-- Heavily inspired by Peter-larsen naive conv2d
-- Idea is to loop over each row of kernels, and then create the output from the row of kernels and
-- all the inputs, using tabulate to get the indexes of the positions
-- probably quite ineffecient, as most papers suggest using im2col approach, but is this naive ?

def conv2d_flatten [l][m][n][o][k] (inputs : [l][m][n]f64) (kernels : [o][l][k][k]f64) =
    let out_m = m-k+1
    let out_n = n-k+1
    let flatD = k * k
    in map(\ kernel3 ->
            tabulate_2d out_m out_n (\ y x ->
                reduce (+) 0 ( flatten (map2 (\ kernel inp ->
                (flatten (map2 (\irow krow -> map2 (*) irow krow)
                    (inp[y:(y+k),x:(x+k)] :> [k][k]f64) kernel)) :> [flatD]f64
                    ) kernel3 inputs
                )
            )
        )
    ) kernels

-- def conv2d_reduce

def conv2d_maps [l][m][n][o][k] (inputs : [l][m][n]f64) (kernels : [l][o][k][k]f64) =
    let out_m = m-k+1
    let out_n = n-k+1
    in tabulate o (\out_channel ->
        tabulate_2d out_m out_n (\outY outX ->
            reduce (+) 0 (tabulate l (\in_channel ->
               reduce (+) 0  (map (reduce (+) 0) (tabulate_2d k k (\i j ->
                    inputs[in_channel,outY+i,outX+j] * kernels[in_channel,out_channel,i,j]
                    ))))
            
            )
        )
    )




def map_ inp fun = map fun inp
def map2_ inp1 inp2 fun = map2 fun inp1 inp2

def conv2dtab3d [l][m][n][o][k] (inputs : [l][m][n]f32) (kernels : [l][o][k][k]f32) =
  let out_m = m-k+1
  let out_n = n-k+1
  in 
  tabulate_3d o out_m out_n (\out_channel y x -> 
    let in_channel_accs = tabulate l (\in_channel ->
      let input_slice = inputs[in_channel, y:(y+k),x:(x+k)] :> [k][k]f32
      let kernels_slice = kernels[in_channel,out_channel] :> [k][k]f32
      in
      let row_accs = map2_ input_slice kernels_slice (\in_row kernel_row ->
        let multiplied_row = map2_ in_row kernel_row (\in_val kernel_val -> 
          in_val * kernel_val
        )
        in reduce (+) 0 multiplied_row
      )
      in reduce (+) 0 row_accs
    )
    in reduce (+) 0 in_channel_accs
  )

def tabulate3d_ (dim1) (dim2) (dim3) (f) =
    tabulate_2d dim1 dim2 (\i j ->
    #[sequential_inner]
     map (\k -> f i j k) (iota dim3) )

def conv2d [l][m][n][o][k] (inputs : [l][m][n]f32) (kernels : [l][o][k][k]f32) =
  let out_m = m-k+1
  let out_n = n-k+1
  in
  tabulate3d_ o (out_m) out_n (\out_channel y x -> 
    reduce (+) 0 (
      map_ (iota l) (\in_channel ->
        reduce (+) 0 (
          map2_ (inputs[in_channel, y:(y+k),x:(x+k)] :> [k][k]f32) (kernels[in_channel,out_channel] :> [k][k]f32) (\in_row kernel_row ->
            reduce (+) 0 (
              map2_ in_row kernel_row (\in_val kernel_val -> 
                in_val * kernel_val
              )
            )
          )
        )
      )
    )
  )


def conv2d_v2 [num_in_ch][in_height][in_width][num_out_ch] (k : i64) (inputs : [num_in_ch][in_height][in_width]f32) (kernels : [num_in_ch][num_out_ch][k][k]f32) =
  let out_height = in_height-k+1
  let out_width = in_width-k+1
  in
  tabulate3d_ num_out_ch (out_height) out_width (\out_channel y x -> 
    reduce (+) 0 (
      map2_ inputs kernels (\input kernel ->
        reduce (+) 0 (
          map2_ (input[y:(y+k),x:(x+k)] :> [k][k]f32) (kernel[out_channel] :> [k][k]f32) (\in_row kernel_row ->
            reduce (+) 0 (
              map2_ in_row kernel_row (\in_val kernel_val -> 
                in_val * kernel_val
              )
            )
          )
        )
      )
    )
  )




-- testing
def validate_small (in_height : i64) (in_width : i64) (in_channels : i64) (out_channels : i64) (radius : i64)  =
    let ker_size = radius * 2 +1
    let inputs = tabulate_3d in_channels in_height in_width (\_ _ _->1)
    let kernels = tabulate_3d in_channels out_channels ker_size (\_ _ _ -> replicate ker_size 1)
    in conv2d inputs kernels

-- main, radius is represented in k, which should be k=2*r+1

-- Bench and small testing
-- ==
-- compiled input @ data/dataset_ker.in
-- auto output
-- compiled random input { [16][512][512]f32 [16][16][5][5]f32 } auto output
-- nobench input @ dataset2.txt
-- output @ small_validate
-- nobench input @ dataset.txt
-- output @ small_validate2
-- compiled input @ data/dataset_small.in
-- output @ modified.txt

def main [num_in_ch][in_height][in_width][num_out_ch] (inputs : [num_in_ch][in_height][in_width]f32) (kernels : [num_in_ch][num_out_ch][5][5]f32) = 
  conv2d inputs kernels
