`bench-final.cu`: tf_nemo_conv + tf_nemo_scan at Super's Mamba shape (128 heads, head dim 64, 8 groups, state 128,
RMAX 32), 40 layers' buffers, min of 21 alternated reps. `a.cu`, `b.cu`, `t.cu` are `zig/kernels/cuda/nemotron_mamba.cu`
at the base 39993136, the previous tip 480ad61f and the tip fa2af585 with `tf_nemo_` renamed to `a_`, `b_`, `t_`.
Built with the glue kernels' flags: `nvcc -std=c++17 -arch=sm_121 -O3 --fmad=false --ftz=false --prec-div=true
--prec-sqrt=true`. The base kernel takes at most 32 kept and window rows, so its cells past that read 0.000 (not run).
