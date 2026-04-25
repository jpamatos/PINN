from typing import Any

import torch
import torch.nn as nn

from data.burgers_data import BurgersTensors

from .base_pinn import BasePINN
from .utils import calculate_grad


class BurgersPINN(BasePINN, nn.Module):
    """PINN implementation for the Burgers equation.

    Equation: u_t + u*u_x = nu*u_xx

    Attributes:
        nu (torch.Tensor): Kinematic viscosity parameter.
    """

    nu: torch.Tensor

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_units: int,
        nu: float,
        activation: str,
    ) -> None:
        """Initializes the BurgersPINN.

        Args:
            num_hidden_layers (int): Number of hidden layers in the MLP.
            hidden_units (int): Number of units per hidden layer.
            nu (float): Kinematic viscosity parameter.
            activation (str): The name of the activation function to use.
        """
        super().__init__(
            input_dim=2,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
        )
        self.register_buffer("nu", torch.tensor(nu))

    def pde_residual(self, data: BurgersTensors) -> torch.Tensor:
        """Computes the PDE residual for the Burgers equation.

        residual = u_t + u*u_x - nu*u_xx

        Args:
            data (BurgersTensors): Container with collocation points.

        Returns:
            torch.Tensor: Mean squared PDE residual.
        """
        x, t = data.x_f, data.t_f
        u = self.forward(torch.cat([x, t], dim=1))
        u_x = calculate_grad(u, x)
        u_t = calculate_grad(u, t)
        u_xx = calculate_grad(u_x, x)
        residual = u_t + u * u_x - self.nu * u_xx
        return torch.mean(residual**2)

    def inital_conditions(self, criterion: Any, data: BurgersTensors) -> torch.Tensor:
        """Computes the initial conditions loss for the Burgers equation.

        Loss IC: u(x, 0) = -sin(pi * x)

        Args:
            criterion (Any): Loss function.
            data (BurgersTensors): Container with initial condition points and values.

        Returns:
            torch.Tensor: Initial condition loss.
        """
        u = self(torch.cat([data.x_i, data.t_i], dim=1))
        return criterion(u, data.u_i)

    def boundary_conditions(self, criterion: Any, data: BurgersTensors) -> torch.Tensor:
        """Computes the boundary conditions loss for the Burgers equation.

        Loss BC: u(-1, t) = 0, u(1, t) = 0

        Args:
            criterion (Any): Loss function.
            data (BurgersTensors): Container with boundary condition points and values.

        Returns:
            torch.Tensor: Boundary condition loss.
        """
        u_b1 = self(torch.cat([data.x_b1, data.t_b], dim=1))
        u_b2 = self(torch.cat([data.x_b2, data.t_b], dim=1))
        return criterion(u_b1, data.u_b) + criterion(u_b2, data.u_b)

    def evaluate(self, data_loader: Any) -> None:
        """Evaluates the model and prints metrics.

        Args:
            data_loader (Any): Data loader for problem data.
        """
        data = data_loader.load()
        self.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            loss_ic = self.inital_conditions(criterion, data)
            loss_bc = self.boundary_conditions(criterion, data)
        loss_pde = self.pde_residual(data)

        print("Burgers PINN Evaluation:")
        print(f"  IC Loss:  {loss_ic.item():.6e}")
        print(f"  BC Loss:  {loss_bc.item():.6e}")
        print(f"  PDE Loss: {loss_pde.item():.6e}")

    def static_params(self) -> dict[str, Any]:
        """Gets the static parameters of the model.

        Returns:
            dict[str, Any]: Dictionary containing static parameters like 'nu'.
        """
        return {"nu": self.nu}
