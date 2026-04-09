from dataclasses import dataclass

import numpy as np
import torch

from data.base_data import BaseLoader


@dataclass
class BurgersTensors:
    """Container for Burgers equation training tensors.

    Attributes:
        x_f (torch.Tensor): Spatial coordinates of collocation points.
        t_f (torch.Tensor): Temporal coordinates of collocation points.
        x_i (torch.Tensor): Spatial coordinates of initial condition points.
        t_i (torch.Tensor): Temporal coordinates of initial condition points.
        u_i (torch.Tensor): Values of the solution at initial condition points.
        x_b1 (torch.Tensor): Spatial coordinates of the first boundary condition points.
        x_b2 (torch.Tensor): Spatial coordinates of the second boundary condition points.
        t_b (torch.Tensor): Temporal coordinates of boundary condition points.
        u_b (torch.Tensor): Values of the solution at boundary condition points.
    """

    x_f: torch.Tensor
    t_f: torch.Tensor
    x_i: torch.Tensor
    t_i: torch.Tensor
    u_i: torch.Tensor
    x_b1: torch.Tensor
    x_b2: torch.Tensor
    t_b: torch.Tensor
    u_b: torch.Tensor


class BurgersLoader(BaseLoader):
    """Data loader for generating training tensors for the Burgers equation.

    Uses the logic extracted from the Burgers' equation notebook.

    Attributes:
        n_f (int): Number of collocation points for the PDE residual.
        n_i (int): Number of initial condition points.
        n_b (int): Number of boundary condition points.
        x_range (tuple[float, float]): Spatial domain range.
        t_range (tuple[float, float]): Temporal domain range.
    """

    def __init__(
        self,
        n_f: int,
        n_i: int,
        n_b: int,
        x_range: tuple[float, float],
        t_range: tuple[float, float],
        device: str | torch.device = "cpu",
    ) -> None:
        """Initializes the BurgersLoader.

        Args:
            n_f (int): Number of collocation points for the PDE residual.
            n_i (int): Number of initial condition points.
            n_b (int): Number of boundary condition points.
            x_range (tuple[float, float]): Spatial domain range.
            t_range (tuple[float, float]): Temporal domain range.
            device (str | torch.device, optional): Device to place tensors on. Defaults to "cpu".
        """
        super().__init__(device=device)
        self.n_f = n_f
        self.n_i = n_i
        self.n_b = n_b
        self.x_range = x_range
        self.t_range = t_range

    def load(self) -> BurgersTensors:
        """Generates and returns the BurgersTensors container.

        Returns:
            BurgersTensors: The generated tensors for the Burgers equation.
        """
        # Collocation points (PDE residual points)
        x_f = np.random.uniform(self.x_range[0], self.x_range[1], self.n_f)
        t_f = np.random.uniform(self.t_range[0], self.t_range[1], self.n_f)

        # Initial condition: u(x, 0) = -sin(pi * x)
        x_i = np.random.uniform(self.x_range[0], self.x_range[1], self.n_i)
        t_i = np.zeros_like(x_i)
        u_i = -np.sin(np.pi * x_i)

        # Boundary conditions: u(-1, t) = 0, u(1, t) = 0
        t_b = np.random.uniform(self.t_range[0], self.t_range[1], self.n_b)
        x_b1 = np.full_like(t_b, self.x_range[0])
        x_b2 = np.full_like(t_b, self.x_range[1])
        u_b = np.zeros_like(t_b)

        return BurgersTensors(
            x_f=self._to_tensor(x_f, requires_grad=True),
            t_f=self._to_tensor(t_f, requires_grad=True),
            x_i=self._to_tensor(x_i),
            t_i=self._to_tensor(t_i),
            u_i=self._to_tensor(u_i),
            x_b1=self._to_tensor(x_b1),
            x_b2=self._to_tensor(x_b2),
            t_b=self._to_tensor(t_b),
            u_b=self._to_tensor(u_b),
        )
