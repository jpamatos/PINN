# Physics-Informed Neural Networks (PINN)

This repository contains implementations of Physics-Informed Neural Networks (PINNs) to solve various partial differential equations (PDEs) using deep learning.

## Overview

The project leverages PyTorch to solve physical problems by incorporating the underlying differential equations into the loss function of neural networks. Experiment tracking and metrics are managed using MLflow, and project configurations are managed by Hydra.

## Implemented Problems

### 1. Burgers' Equation (1D)
Solves the 1D Burgers' equation, a fundamental PDE in fluid mechanics:
$$u_t + u\,u_x = \nu\,u_{xx}$$
**Conditions:**
- Initial: $u(x, 0) = -\sin(\pi x)$
- Boundary: $u(-1, t) = u(1, t) = 0$

### 2. Heat Equation (2D)
Solves the 2D Heat equation on a square domain:
$$u_t = u_{xx} + u_{yy}$$
**Conditions:**
- Initial: $u(x,y,0) = \sin(\pi x)\sin(\pi y)$
- Boundary: Zero Dirichlet conditions on all edges ($x=\pm 1, y=\pm 1$)

### 3. Navier-Stokes Equations
Solves the momentum and continuity equations for incompressible flow:
- $u_t + u\,u_x + v\,u_y = -p_x + \frac{1}{Re}(u_{xx} + u_{yy})$
- $v_t + u\,v_x + v\,v_y = -p_y + \frac{1}{Re}(v_{xx} + v_{yy})$
- $u_x + v_y = 0$

**Conditions:**
- Initial: Taylor-Green vortex setup ($u(x,y,0) = \sin(x)\cos(y)$, $v(x,y,0) = -\cos(x)\sin(y)$, $p(x,y,0) = -\frac{1}{4}(\cos(2x) + \cos(2y))$)
- Boundary: Periodic boundary conditions on domain $[0, 2\pi]$

## Running Experiments

This project uses `uv` for dependency management and `hydra` for configuration. The main entry point is `src/run.py`.

### Prerequisites

Ensure you have `uv` installed. You can install project dependencies into the virtual environment using:
```bash
uv sync
```

### Execution

To run the default experiment (Navier-Stokes equation):
```bash
uv run src/run.py
```

To modify the training parameters, you can use Hydra overrides directly from the command line:
```bash
uv run src/run.py trainer.epochs=1000 trainer.lr=0.005
```

### Changing the PDE Problem

The configuration is defined via dataclasses in `src/configs.py`. By default, `StokesPINNConfig` and `StokesDataConfig` are used. 

To run a different problem (e.g., Burgers' equation), the simplest way is to modify the default factories of the `Config` dataclass in `src/configs.py`:

```python
@dataclass
class Config:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Any = field(default_factory=BurgersPINNConfig) # Update here
    data: Any = field(default_factory=BurgersDataConfig)  # Update here
```

### Viewing Results

The project uses MLflow for tracking. To view the training logs and metrics (including IC Loss, BC Loss, PDE Loss, and specific relative errors calculated in the evaluation loops):
```bash
uv run mlflow ui --backend-store-uri mlruns
```
Then navigate to `http://localhost:5000` in your web browser.
