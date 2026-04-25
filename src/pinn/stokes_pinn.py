from typing import Any

import torch
import torch.nn as nn

from data.stokes_data import StokesTensors

from .base_pinn import BasePINN
from .utils import calculate_grad


class StokesPINN(BasePINN, nn.Module):
    """PINN implementation for the Navier-Stokes equations.

    Attributes:
        nu (torch.Tensor): Kinematic viscosity parameter.
        pi (torch.Tensor): Pi constant for normalization.
        T (torch.Tensor): Max time for normalization.
    """

    nu: torch.Tensor
    pi: torch.Tensor
    T: torch.Tensor

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_units: int,
        nu: float,
        activation: str,
    ) -> None:
        """Initializes the StokesPINN.

        Args:
            num_hidden_layers (int): Number of hidden layers in the MLP.
            hidden_units (int): Number of units per hidden layer.
            nu (float): Kinematic viscosity parameter.
            activation (str): The name of the activation function to use.
        """
        super().__init__(
            input_dim=3,
            output_dim=3,
            num_hidden_layers=num_hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
        )
        self.register_buffer("nu", torch.tensor(nu))
        self.register_buffer("pi", torch.tensor(torch.pi))
        self.register_buffer("T", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network with normalization.

        Args:
            x (torch.Tensor): The input tensor of shape (N, 3) containing (x, y, t).

        Returns:
            torch.Tensor: The output of the neural network (u, v, p).
        """
        x_coord, y_coord, t_coord = torch.split(x, 1, dim=1)
        x_n = (x_coord - self.pi) / self.pi
        y_n = (y_coord - self.pi) / self.pi
        t_n = t_coord / self.T
        return self.net(torch.cat([x_n, y_n, t_n], dim=1))

    def pde_residual(self, data: StokesTensors) -> torch.Tensor:
        """Computes the PDE residual for the Navier-Stokes equations.

        Args:
            data (StokesTensors): Container with collocation points.

        Returns:
            torch.Tensor: Mean squared PDE residual.
        """
        x, y, t = data.x_f, data.y_f, data.t_f
        out = self.forward(torch.cat([x, y, t], dim=1))
        u, v, p = torch.split(out, 1, dim=1)

        u_t = calculate_grad(u, t)
        u_x = calculate_grad(u, x)
        u_y = calculate_grad(u, y)
        u_xx = calculate_grad(u_x, x)
        u_yy = calculate_grad(u_y, y)
        p_x = calculate_grad(p, x)
        res_u = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)

        v_t = calculate_grad(v, t)
        v_x = calculate_grad(v, x)
        v_y = calculate_grad(v, y)
        v_xx = calculate_grad(v_x, x)
        v_yy = calculate_grad(v_y, y)
        p_y = calculate_grad(p, y)
        res_v = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)

        res_c = u_x + v_y

        return torch.mean(res_u**2 + res_v**2 + res_c**2)

    def inital_conditions(self, criterion: Any, data: StokesTensors) -> torch.Tensor:
        """Computes the initial conditions loss for the Navier-Stokes equations.

        Args:
            criterion (Any): Loss function.
            data (StokesTensors): Container with initial condition points and values.

        Returns:
            torch.Tensor: Initial condition loss.
        """
        pred_ic = self(torch.cat([data.x_i, data.y_i, data.t_i], dim=1))
        pred_ui, pred_vi, pred_pi = torch.split(pred_ic, 1, dim=1)

        return (
            criterion(pred_ui, data.u_i)
            + criterion(pred_vi, data.v_i)
            + criterion(pred_pi, data.p_i)
        )

    def boundary_conditions(self, criterion: Any, data: StokesTensors) -> torch.Tensor:
        """Computes the boundary conditions loss for the Navier-Stokes equations.

        Args:
            criterion (Any): Loss function.
            data (StokesTensors): Container with boundary condition points and values.

        Returns:
            torch.Tensor: Boundary condition loss.
        """
        pred_b_l = self(torch.cat([data.x_b_l, data.y_b_x, data.t_b_x], dim=1))
        pred_b_r = self(torch.cat([data.x_b_r, data.y_b_x, data.t_b_x], dim=1))
        pred_b_b = self(torch.cat([data.x_b_y, data.y_b_b, data.t_b_y], dim=1))
        pred_b_u = self(torch.cat([data.x_b_y, data.y_b_u, data.t_b_y], dim=1))

        return criterion(pred_b_l, pred_b_r) + criterion(pred_b_b, pred_b_u)

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
            Re = 1.0 / self.nu
            u_exact = torch.sin(x) * torch.cos(y) * torch.exp(-2 * t / Re)
            v_exact = -torch.cos(x) * torch.sin(y) * torch.exp(-2 * t / Re)
            p_exact = (
                -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * torch.exp(-4 * t / Re)
            )

            pred = self(torch.cat([x, y, t], dim=1))
            u_pred, v_pred, p_pred = torch.split(pred, 1, dim=1)

            def l2_relative(pred_t, exact_t):
                return torch.norm(pred_t - exact_t) / torch.norm(exact_t)

            err_u = l2_relative(u_pred, u_exact)
            err_v = l2_relative(v_pred, v_exact)
            err_p = l2_relative(p_pred, p_exact)

        loss_pde = self.pde_residual(data)

        print("Navier-Stokes PINN Evaluation:")
        print(f"  IC Loss:  {loss_ic.item():.6e}")
        print(f"  BC Loss:  {loss_bc.item():.6e}")
        print(f"  PDE Loss: {loss_pde.item():.6e}")
        print(f"  L2 Rel Error u: {err_u.item():.6e}")
        print(f"  L2 Rel Error v: {err_v.item():.6e}")
        print(f"  L2 Rel Error p: {err_p.item():.6e}")

    def static_params(self) -> dict[str, Any]:
        """Gets the static parameters of the model.

        Returns:
            dict[str, Any]: Dictionary containing static parameters like 'nu'.
        """
        return {"nu": self.nu}
