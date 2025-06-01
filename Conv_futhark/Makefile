

datasets:
		mkdir -p data
		futhark dataset -g [16][128][128]f32 --f32-bounds=0.0:2.0 -g [16][16][5][5]f32 > data/dataset_small.in
		futhark dataset -g [8][132][132]f32 --f32-bounds=0.0:2.0 -g [8][16][5][5]f32 > data/dataset_emil.in
		futhark dataset -g [32][1028][1028]f32 --f32-bounds=0.0:2.0 -g [32][32][5][5]f32 > data/dataset_med.in
		futhark dataset -g [64][2052][2052]f32 --f32-bounds=0.0:2.0 -g [64][64][5][5]f32 > data/dataset_ker.in
		futhark dataset -g [64][1024][1024]f32 --f32-bounds=0.0:2.0 -g [64][64][5][5]f32 > data/dataset_big.in
		futhark dataset -g [64][4096][4096]f32 --f32-bounds=0.0:2.0 -g [64][64][3][3]f32 > data/datset_cudaver.in


run: conv_naive.fut
	futhark bench --backend=cuda conv_naive.fut

test: conv_naive.fut
	futhark dev -a --backend=cuda conv_naive_test.fut_gpu
	./conv_naive_test < data/dataset_small.in > modified.txt
	futhark test --backend=cuda conv_naive.fut

clean:
	rm -rf *.actual *.expected
	rm -rf *.c conv_naive