import numpy as np

def organics(t, y, params, attention_fn, stimulus_fn):

    N = params.N

    v = y[:N]
    a = y[N:2*N]
    u = y[2*N:3*N]

    x = stimulus_fn(t)  
    b = attention_fn(t) 

    if params.Wzx is None:
        z = x 
    else:
        z = params.Wzx @ x 

    y_tmp = np.maximum(v, 0) ** 2

    y_hat = params.Wyy @ np.sqrt(y_tmp)

    gain_b = b / (1 + b)
    gain_a = 1.0 / (1.0 + a)
    
    dv = (-v + gain_b * z + gain_a * y_hat) / params.tau_v

    u_weighted = params.Wnorm @ (y_tmp * u)

    c = ((params.sigma * params.b0) / (1.0 + params.b0)) ** 2
    
    du = (-u + u_weighted + c) / params.tau_u

    sqrt_u = np.sqrt(np.maximum(u, 0))
    da = (-a + sqrt_u + a * sqrt_u) / params.tau_a

    out = np.empty(3 * N)
    out[:N]    = dv
    out[N:2*N] = da
    out[2*N:]  = du
    return out
