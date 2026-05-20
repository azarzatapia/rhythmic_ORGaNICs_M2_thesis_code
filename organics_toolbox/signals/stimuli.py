import numpy as np

def orientation_stimulus(onset, offset, angle_deg, amplitude, theta_axis):

    theta_axis = np.asarray(theta_axis, dtype=float)
    idx = int(np.argmin(np.abs(theta_axis - float(angle_deg))))

    stim_vec = np.zeros_like(theta_axis, dtype=float)
    stim_vec[idx] = float(amplitude)

    return lambda t: stim_vec if onset <= t <= offset else np.zeros_like(stim_vec)


def spatial_stimulus(onset, offset, center_deg, size_deg, amplitude, xi_axis):

    xi_axis = np.asarray(xi_axis, dtype=float)

    mask = np.abs(xi_axis - float(center_deg)) <= 0.5 * float(size_deg)
    stim_vec = float(amplitude) * mask.astype(float)

    return lambda t: stim_vec if (onset <= t <= offset) else np.zeros_like(stim_vec)

def grating_plus_noise_stimulus(onset, offset, angle_deg, theta_axis,
                                 contThresh=0.0, noise_contrast=0.2,
                                 target_present=True, seed=None,
                                 n_components=None):
    theta_axis = np.asarray(theta_axis, float)
    M = theta_axis.size
    rng = np.random.default_rng(seed)

    if n_components is None:
        n_components = max(1, M // 6)
    if n_components > M:
        raise ValueError(f"n_components ({n_components}) must not exceed M ({M})")

    noise_vec = np.zeros(M, dtype=float)
    noise_vec[rng.choice(M, size=n_components, replace=False)] = float(noise_contrast)

    grating_vec = np.zeros(M, dtype=float)
    if target_present:
        grating_vec[int(np.argmin(np.abs(theta_axis - float(angle_deg))))] = float(contThresh)

    x    = np.clip(0.5 * (noise_vec + grating_vec), 0.0, 1.0)
    zero = np.zeros_like(x)
    return lambda t: x if onset <= t <= offset else zero