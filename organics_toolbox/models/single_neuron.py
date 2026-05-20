import numpy as np
from organics_toolbox.core.params import FullParams
from .base import ORGaNICsBaseModel

class SingleNeuronModel(ORGaNICsBaseModel):
    """
    A minimal 1-neuron ORGaNICs model.
    Good for debugging dynamics, tuning parameters, etc.
    """

    def __init__(
        self,
        tau_v=2,
        tau_a=1,
        tau_u=2,
        sigma=0.1,
        b0=0.5,
        w_in=1.0,
        w_rec=0.0,
    ):
        Wzx = np.array([[w_in]])      # input→neuron
        Wyy = np.array([[w_rec]])     # recurrent
        Wnorm = np.array([[1.0]])     # trivial norm

        params = FullParams(
            N=1,
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
