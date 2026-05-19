from enum import Enum, auto
from pathlib import Path

import numpy as np
import torch

from abc import ABC, abstractmethod

from dataclasses import dataclass


# marker classes to preserve some level of typing

class SubsolverCfg(ABC):
    pass

class ConstrSolverCfg(ABC):
    pass


# resulting output classes

class SubsolverCriterion(Enum):
    DISTANCE = auto()
    NORM = auto()

@dataclass
class SubsolverHistory:
    p_hist: torch.Tensor
    f_hist: torch.Tensor


@dataclass
class SubsolverResult:
    success: bool
    p: torch.Tensor
    iters: int
    history: SubsolverHistory

class CustomSubsolverResult(ABC):
    @property
    @abstractmethod
    def result(self) -> SubsolverResult:
        pass


@dataclass
class ConstrSolverHistory:
    p_hist: torch.Tensor  # position history

    f_hist: torch.Tensor  # cost function
    gs_hist: torch.Tensor
    hs_hist: torch.Tensor

    g_mults_hist: torch.Tensor
    h_mults_hist: torch.Tensor


@dataclass
class ConstrSolverResult:
    success: bool
    p: torch.Tensor  # final position
    iters: int
    history: ConstrSolverHistory


class CustomConstrSolverResult(ABC):
    @property
    @abstractmethod
    def result(self) -> ConstrSolverResult:
        pass
