# Non-Linear Data Augmentation

__TODO__
1. Interpolate the source image instead of rounding phi
2. Compute the $v_i$ via the kernel framework instead of interpolating. This needs to happen in each step of the forward Euler.
3. Look into using PyTorch or Tensorflow for parallelization of the matrix computations. Probably PyTorch is preferable, only concern is that sparse arrays are in beta only and our code is crucially dependent on these at the moment. Measure performance via timing.
4. Make test case: Make random alpha, apply transformation (here: "dense"/full resolution application needs to be implemented).
5. Treat the alpha as m x 3 arrays instead of 3m x 1 arrays. Then, the S matrix doesn't need to be blown up which is slow. Some things will need to be reshaped as a consequence of this.
6. (Related to 5.) Flattening out the image gradent x-,y- and z-dimensions in order to put it into diagonal matrix is a huge bottleneck for high-res evaluation. This should be rewritten, maybe similarly to 5.? (function gradient.dIm_dphi)
