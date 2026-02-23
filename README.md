# research-geo-diff-opt-layer

> TODO: flesh this out further before public release

Differentiable manifold optimization library can be found in the `mfld_optim` folder. For training and results, refer to the directory `training`. There are four (4) subdirectories, each a different variant according to the following:

- `nn_and_euclid_cbf_layer`: trained controller and a separate Euclidean CBF safety layer not included in backpropagation (using `cvxpy`)
- `nn_with_euclid_cbf_layer`: trained controller with Euclidean CBF safety layer included in backpropagation (using `cvxpy`)
- `nn_with_gcbf_layer`: trained controller using Geometric CBF (GCBF) safety layer included in backpropagation
- `nn_with_prod_gcbf_layer`: trained controller using product formulation of GCBF safety layer included in backpropagation

## Custom Python Installation

Unfortunately as this is testing code the differentiable geometric optimization layers are writtenin pure Python and is slow as a result. Using process based `multiprocessing` is not possible due to an error with the way some `torch` options are serialized so we have to use threading. Therefore to get a large speedup we need a vrsion of Python released from GIL.

```bash
sudo add-apt-repository ppa:deadsnakes
sudo apt-get update
sudo apt-get install python3.14-nogil python3.14 python3.14-dev
```

Now we need to setup the virtual environment with this free-threading build.

```bash
sudo apt install python3.14-venv

python3.14-nogil -m venv .venv
source .venv/bin/activate
```

Check that the environment is setup properly.

```python
import sys
sys._is_gil_enabled()  # should be False
```
