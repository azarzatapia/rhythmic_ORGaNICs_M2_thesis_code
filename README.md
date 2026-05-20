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
from organics_toolbox.signals.attention import windowed_attention
import matplotlib.pyplot as plt
import numpy as np

model = SingleNeuronModel()

theta_axis = np.array([0.0])
stim = orientation_stimulus(onset=100, offset=400, angle_deg=0, amplitude=1.0, theta_axis=theta_axis)
att  = windowed_attention(N=1, onset=0, offset=500, gain=0.2)

t, v = model.simulate(stim, att, t_span=(0, 500), dt=1.0)
y = np.maximum(v[0, :], 0) ** 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, y, label='Response')
ax.set_xlabel('Time')
ax.set_ylabel('Response (a.u.)')
ax.set_title('Single Neuron Response to Stimulus')
ax.legend()
plt.tight_layout()
plt.show()
```
