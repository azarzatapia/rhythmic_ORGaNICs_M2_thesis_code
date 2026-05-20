from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FullParams:
 
    N: int                 # number of neurons
    Wzx: np.ndarray        # input → population
    Wyy: np.ndarray        # recurrent → population
    Wnorm: np.ndarray      # normalization matrix

    tau_v: float
    tau_a: float
    tau_u: float

    sigma: float
    b0: float

    Wxi_norm: Optional[np.ndarray] = None 
    Ntheta_k: Optional[int] = None
    Nxi_k:    Optional[int] = None

    def validate(self):
        assert self.Wzx.shape[0] == self.N
        assert self.Wyy.shape == (self.N, self.N)
        assert self.Wnorm.shape == (self.N, self.N)