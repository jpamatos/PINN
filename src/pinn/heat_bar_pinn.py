from typing import Any

import torch
import torch.nn as nn

from data.heat_bar_data import HeatBarTensors

from .base_pinn import BasePINN
from .utils import calculate_grad


class HeatBarPINN(BasePINN):
    """PINN implementation for the Heat equation on a 2D bar.

    Equation: u_t = u_xx + u_yy
    """

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_units: int,
        activation: str,
    ) -> None:
        """Initializes the HeatBarPINN.

        Args:
            num_hidden_layers (int): Number of hidden layers in the MLP.
            hidden_units (int): Number of units per hidden layer.
            activation (str): The name of the activation function to use.
        """
        super().__init__(
            input_dim=3,
            output_dim=1,
            num_hidden_layers=num_hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
        )

    def pde_residual(self, data: HeatBarTensors) -> torch.Tensor:
        """Computes the PDE residual for the Heat equation.

        residual = u_t - u_xx - u_yy

        Args:
            data (HeatBarTensors): Container with collocation points.

        Returns:
            torch.Tensor: Mean squared PDE residual.
        """
        x, y, t = data.x_f, data.y_f, data.t_f
        u = self.forward(torch.cat([x, y, t], dim=1))

        u_t = calculate_grad(u, t)
        u_x = calculate_grad(u, x)
        u_y = calculate_grad(u, y)
        u_xx = calculate_grad(u_x, x)
        u_yy = calculate_grad(u_y, y)

        residual = u_t - u_xx - u_yy
        return torch.mean(residual**2)

    def inital_conditions(self, criterion: Any, data: HeatBarTensors) -> torch.Tensor:
        """Computes the initial conditions loss.

        Loss IC: u(x, y, 0) = sin(pi * x) * sin(pi * y)

        Args:
            criterion (Any): Loss function.
            data (HeatBarTensors): Container with initial condition points and values.

        Returns:
            torch.Tensor: Initial condition loss.
        """
        u = self(torch.cat([data.x_i, data.y_i, data.t_i], dim=1))
        return criterion(u, data.u_i)

    def boundary_conditions(self, criterion: Any, data: HeatBarTensors) -> torch.Tensor:
        """Computes the boundary conditions loss.

        Loss BC: u(-1, y, t) = 0, u(1, y, t) = 0, u(x, -1, t) = 0, u(x, 1, t) = 0

        Args:
            criterion (Any): Loss function.
            data (HeatBarTensors): Container with boundary condition points and values.

        Returns:
            torch.Tensor: Boundary condition loss.
        """
        u_bl = self(torch.cat([data.x_bl, data.y_bl, data.t_b], dim=1))
        u_br = self(torch.cat([data.x_br, data.y_br, data.t_b], dim=1))
        u_bu = self(torch.cat([data.x_bu, data.y_bu, data.t_b], dim=1))
        u_bd = self(torch.cat([data.x_bd, data.y_bd, data.t_b], dim=1))

        loss_bl = criterion(u_bl, data.u_b)
        loss_br = criterion(u_br, data.u_b)
        loss_bu = criterion(u_bu, data.u_b)
        loss_bd = criterion(u_bd, data.u_b)

        return loss_bl + loss_br + loss_bu + loss_bd

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

            x, y, t = data.x_f, data.y_f, data.t_f
            u_exact = (
                torch.exp(-2 * (torch.pi**2) * t)
                * torch.sin(torch.pi * x)
                * torch.sin(torch.pi * y)
            )
            u_pred = self(torch.cat([x, y, t], dim=1))
            rmse = torch.sqrt(torch.mean((u_pred - u_exact) ** 2))

        loss_pde = self.pde_residual(data)

        print("Heat Bar PINN Evaluation:")
        print(f"  IC Loss:  {loss_ic.item():.6e}")
        print(f"  BC Loss:  {loss_bc.item():.6e}")
        print(f"  PDE Loss: {loss_pde.item():.6e}")
        print(f"  RMSE (vs Analytical): {rmse.item():.6e}")

    def static_params(self) -> dict[str, Any]:
        """Gets the static parameters of the model.

        Returns:
            dict[str, Any]: Dictionary containing static parameters (empty for this model).
        """
        return {}
