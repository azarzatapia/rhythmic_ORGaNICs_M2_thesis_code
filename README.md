# Rhythmic ORGaNICs Repository

This repository holds the code used to create the model for the M2 thesis project by André R. Zarza Tapia.

## Repository Structure

```
organics_toolbox/
├── core/
│   ├── organics_dynamics.py
│   ├── params.py
│   ├── matrices.py
│   └── solvers.py
├── models/
│   ├── base.py
│   ├── single_neuron.py
│   ├── population_1d.py
│   └── population_2d.py
└── signals/
    ├── stimuli.py
    ├── attention.py
    └── encoding.py
demo.ipynb
```

## Quick Start

Install in editable mode:

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

Then open `demo.ipynb` for an end-to-end example.

### Minimal example

```python
from organics_toolbox.models.single_neuron import SingleNeuronModel
from organics_toolbox.signals.stimuli import orientation_stimulus
from organics_toolbox.signals.attention import rhythmic_attention
import numpy as np

model = SingleNeuronModel(tau_v=2, tau_a=1, tau_u=2, sigma=0.1, b0=0.5)

theta_axis = np.array([0.0])
stim = orientation_stimulus(onset=100, offset=400, angle_deg=0, amplitude=1.0, theta_axis=theta_axis)
att  = rhythmic_attention(N=1, onset=0, offset=500, amp=0.25, freq_hz=4.0)

t, sol = model.simulate(stim, att, t_span=(0, 500), dt=1e-3)
v = sol[:, 0]   # membrane potential trace
```
