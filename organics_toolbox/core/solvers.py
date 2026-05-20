import numpy as np
from scipy.integrate import solve_ivp

def simulate_ode(rhs, t_span, x0, dt=1e-2, rtol=1e-4, atol=1e-6, verbose=False):
    """
    Integrate an ODE using solve_ivp and return (t, y).

    rhs : function(t, y) → dy/dt
    t_span : (t0, t1)
    x0 : initial state vector
    dt : output sampling step
    """
    t0, t1 = t_span
    t_eval = np.arange(t0, t1, dt)

    sol = solve_ivp(rhs,
                    t_span,
                    x0,
                    t_eval=t_eval,
                    rtol=rtol,
                    atol=atol,
                    method='RK45')
    if verbose:
        if sol.success:
            print(sol.message)
        else :
            raise RuntimeError(f"Simulation failed: {sol.message}")

    return sol.t, sol.y