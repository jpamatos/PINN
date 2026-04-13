# Physics-Informed Neural Networks (PINN)

This repository contains implementations of Physics-Informed Neural Networks (PINNs) to solve various partial differential equations (PDEs) using deep learning.

## Overview

The project leverages PyTorch to solve physical problems by incorporating the underlying differential equations into the loss function of neural networks. Experiment tracking and metrics are managed using MLflow.

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
- Initial: Taylor-Green vortex setup
- Boundary: Periodic boundary conditions on domain $[0, 2\pi]$
