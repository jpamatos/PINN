from dataclasses import dataclass

import numpy as np
import torch

from data.base_data import BaseLoader


@dataclass
class HeatBarTensors:
    """Container for Heat Bar equation training tensors.

    Attributes:
        x_f (torch.Tensor): Spatial x coordinates of collocation points.
        y_f (torch.Tensor): Spatial y coordinates of collocation points.
        t_f (torch.Tensor): Temporal coordinates of collocation points.
        x_i (torch.Tensor): Spatial x coordinates of initial condition points.
        y_i (torch.Tensor): Spatial y coordinates of initial condition points.
        t_i (torch.Tensor): Temporal coordinates of initial condition points.
        u_i (torch.Tensor): Values of the solution at initial condition points.
        t_b (torch.Tensor): Temporal coordinates of boundary condition points.
        x_bl (torch.Tensor): Spatial x coordinates of left boundary condition points.
        y_bl (torch.Tensor): Spatial y coordinates of left boundary condition points.
        x_br (torch.Tensor): Spatial x coordinates of right boundary condition points.
        y_br (torch.Tensor): Spatial y coordinates of right boundary condition points.
        x_bu (torch.Tensor): Spatial x coordinates of upper boundary condition points.
        x_bd (torch.Tensor): Spatial x coordinates of lower boundary condition points.
        y_bu (torch.Tensor): Spatial y coordinates of upper boundary condition points.
        y_bd (torch.Tensor): Spatial y coordinates of lower boundary condition points.
        u_b (torch.Tensor): Values of the solution at boundary condition points.
    """

    x_f: torch.Tensor
    y_f: torch.Tensor
    t_f: torch.Tensor
    x_i: torch.Tensor
    y_i: torch.Tensor
    t_i: torch.Tensor
    u_i: torch.Tensor
    t_b: torch.Tensor
    x_bl: torch.Tensor
    y_bl: torch.Tensor
    x_br: torch.Tensor
    y_br: torch.Tensor
    x_bu: torch.Tensor
    x_bd: torch.Tensor
    y_bu: torch.Tensor
    y_bd: torch.Tensor
    u_b: torch.Tensor


class HeatBarLoader(BaseLoader):
    """Data loader for generating training tensors for the Heat Bar equation.

    Uses the logic extracted from the Heat Bar equation notebook.

    Attributes:
        n_f (int): Number of collocation points for the PDE residual.
        n_i (int): Number of initial condition points.
        n_b (int): Number of boundary condition points.
        x_range (tuple[float, float]): Spatial domain range for x.
        y_range (tuple[float, float]): Spatial domain range for y.
        t_range (tuple[float, float]): Temporal domain range.
    """

    def __init__(
        self,
        n_f: int,
        n_i: int,
        n_b: int,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        t_range: tuple[float, float],
        device: str | torch.device = "cpu",
    ) -> None:
        """Initializes the HeatBarLoader.

        Args:
            n_f (int): Number of collocation points for the PDE residual.
            n_i (int): Number of initial condition points.
            n_b (int): Number of boundary condition points.
            x_range (tuple[float, float]): Spatial domain range for x.
            y_range (tuple[float, float]): Spatial domain range for y.
            t_range (tuple[float, float]): Temporal domain range.
            device (str | torch.device, optional): Device to place tensors on. Defaults to "cpu".
        """
        super().__init__(device=device)
        self.n_f = n_f
        self.n_i = n_i
        self.n_b = n_b
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range

    def load(self) -> HeatBarTensors:
        """Generates and returns the HeatBarTensors container.

        Returns:
            HeatBarTensors: The generated tensors for the Heat Bar equation.
        """
        # Collocation points (PDE residual points)
        x_f = np.random.uniform(self.x_range[0], self.x_range[1], self.n_f)
        y_f = np.random.uniform(self.y_range[0], self.y_range[1], self.n_f)
        t_f = np.random.uniform(self.t_range[0], self.t_range[1], self.n_f)

        # Initial condition: u(x, y, 0) = sin(pi * x) * sin(pi * y)
        x_i = np.random.uniform(self.x_range[0], self.x_range[1], self.n_i)
        y_i = np.random.uniform(self.y_range[0], self.y_range[1], self.n_i)
        t_i = np.zeros_like(x_i)
        u_i = np.sin(np.pi * x_i) * np.sin(np.pi * y_i)

        # Boundary conditions
        t_b = np.random.uniform(self.t_range[0], self.t_range[1], self.n_b)

        # Left boundary (x = x_min)
        x_bl = np.full_like(t_b, self.x_range[0])
        y_bl = np.random.uniform(self.y_range[0], self.y_range[1], self.n_b)

        # Right boundary (x = x_max)
        x_br = np.full_like(t_b, self.x_range[1])
        y_br = np.random.uniform(self.y_range[0], self.y_range[1], self.n_b)

        # Upper boundary (y = y_min)
        x_bu = np.random.uniform(self.x_range[0], self.x_range[1], self.n_b)
        y_bu = np.full_like(x_bu, self.y_range[0])

        # Lower boundary (y = y_max)
        x_bd = np.random.uniform(self.x_range[0], self.x_range[1], self.n_b)
        y_bd = np.full_like(x_bd, self.y_range[1])

        u_b = np.zeros_like(t_b)

        return HeatBarTensors(
            x_f=self._to_tensor(x_f, requires_grad=True),
            y_f=self._to_tensor(y_f, requires_grad=True),
            t_f=self._to_tensor(t_f, requires_grad=True),
            x_i=self._to_tensor(x_i),
            y_i=self._to_tensor(y_i),
            t_i=self._to_tensor(t_i),
            u_i=self._to_tensor(u_i),
            t_b=self._to_tensor(t_b),
            x_bl=self._to_tensor(x_bl),
            y_bl=self._to_tensor(y_bl),
            x_br=self._to_tensor(x_br),
            y_br=self._to_tensor(y_br),
            x_bu=self._to_tensor(x_bu),
            x_bd=self._to_tensor(x_bd),
            y_bu=self._to_tensor(y_bu),
            y_bd=self._to_tensor(y_bd),
            u_b=self._to_tensor(u_b),
        )
