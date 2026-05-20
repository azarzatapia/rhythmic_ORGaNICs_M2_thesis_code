import numpy as np

def windowed_attention(N, onset, offset, gain, baseline=0.5):
    base = np.ones(N) * baseline

    def att(t):
        if onset <= t <= offset:
            return base * gain
        else:
            return base

    return att


def orientation_attention(N, neuron_index, onset, offset, gain, baseline=0.5):
    base = np.ones(N) * baseline

    def att(t):
        arr = base.copy()
        if onset <= t <= offset:
            arr[neuron_index] *= gain 
        return arr

    return att


def rhythmic_attention(
    N,
    onset=0,
    offset=1e9,
    indices=None,
    baseline=0.5,
    amp=0.25,
    freq_hz=4.0,
    phase_shifts=None,
):

    base = np.ones(N) * baseline

    if indices is None:
        indices = np.arange(N)
    else:
        indices = np.asarray(indices)

    if phase_shifts is None:
        phase_shifts = np.zeros(N)

    def att(t):
        arr = base.copy()
        if onset <= t <= offset:
            t_sec = (t - onset) / 1000.0
            angles = 2 * np.pi * freq_hz * t_sec + phase_shifts[indices]
 
            arr[indices] = baseline + amp * np.sin(angles)
        return arr

    return att