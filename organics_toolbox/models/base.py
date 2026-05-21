import numpy as np
from organics_toolbox.core.solvers import simulate_ode
from organics_toolbox.core.organics_dynamics import organics

class ORGaNICsBaseModel:

    def __init__(self, params):
        self.params = params

    def simulate(
        self,
        stimulus_fn,
        attention_fn,
        t_span=(0, 500),
        x0=None,
        dt=1e-3,
        verbose=False,
        method='rk45'
    ):
        N = self.params.N

        if x0 is None:
            x0 = np.concatenate([
                np.zeros(N),
                np.zeros(N),
                np.zeros(N),
            ])

        def rhs(t, y):
            return organics(t, y, self.params, attention_fn, stimulus_fn)
    
        if method == 'rk45':
            return simulate_ode(rhs, t_span, x0, dt=dt, verbose=verbose)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'rk45' or 'euler'.")
