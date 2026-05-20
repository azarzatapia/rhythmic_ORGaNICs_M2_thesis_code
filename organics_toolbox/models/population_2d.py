import numpy as np
from organics_toolbox.core.params import FullParams
from .base import ORGaNICsBaseModel

class Population2DModel(ORGaNICsBaseModel):

    def __init__(self, n_ori, n_space, Wyy, Wnorm, Wzx=None,
                 tau_v=2, tau_a=1, tau_u=2,
                 sigma=0.1, b0=0.5,
                 Wxi_norm=None):
        
        N = n_ori * n_space

        if Wyy is None or Wnorm is None:
            raise ValueError(
                "Population2DModel requires Wyy and Wnorm. "
                "Use MakeMatrices to build them."
            )


        params = FullParams(
            N=N,
            Wzx=Wzx,
            Wyy=Wyy,
            Wnorm=Wnorm,
            tau_v=tau_v,
            tau_a=tau_a,
            tau_u=tau_u,
            sigma=sigma,
            b0=b0,
            Wxi_norm=Wxi_norm,
            Ntheta_k=n_ori  if Wxi_norm is not None else None,
            Nxi_k=n_space   if Wxi_norm is not None else None,
        )

        super().__init__(params)
