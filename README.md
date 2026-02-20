# research-geo-diff-opt-layer

> TODO: flesh this out further before public release

Differentiable manifold optimization library can be found in the `mfld_optim` folder. For training and results, refer to the directory `training`. There are four (4) subdirectories, each a different variant according to the following:

- `nn_and_euclid_cbf_layer`: trained controller and a separate Euclidean CBF safety layer not included in backpropagation (using `cvxpy`)
- `nn_with_euclid_cbf_layer`: trained controller with Euclidean CBF safety layer included in backpropagation (using `cvxpy`)
- `nn_with_gcbf_layer`: trained controller using Geometric CBF (GCBF) safety layer included in backpropagation
- `nn_with_prod_gcbf_layer`: trained controller using product formulation of GCBF safety layer included in backpropagation
