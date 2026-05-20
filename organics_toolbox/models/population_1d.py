from organics_toolbox.core.params import FullParams
from .base import ORGaNICsBaseModel

class Population1DModel(ORGaNICsBaseModel):

    def __init__(self, n_units, Wzx, Wyy, Wnorm,
                 tau_v=2, tau_a=1, tau_u=2,
                 sigma=0.1, b0=0.5):

        N = n_units

        if Wzx is None or Wyy is None or Wnorm is None:
            raise ValueError(
                "Population1DModel requires Wzx, Wyy, and Wnorm. "
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
        )

        super().__init__(params)
