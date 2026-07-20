import rootutils

_PROJ_ROOT = rootutils.setup_root(search_from=__file__, indicator=".project-root")
CVXPY_RESULTS = _PROJ_ROOT / "offline_training/results/cvxpy"
DMOL_RESULTS = _PROJ_ROOT / "offline_training/results/dmol"
