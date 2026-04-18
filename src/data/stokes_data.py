from dataclasses import dataclass

import numpy as np
import torch

from data.base_data import BaseLoader


@dataclass
class StokesTensors:
    """Container for Stokes equation training tensors.

    Attributes:
        x_f (torch.Tensor): Spatial x coordinates of collocation points.
        y_f (torch.Tensor): Spatial y coordinates of collocation points.
        t_f (torch.Tensor): Temporal coordinates of collocation points.
        x_i (torch.Tensor): Spatial x coordinates of initial condition points.
        y_i (torch.Tensor): Spatial y coordinates of initial condition points.
        t_i (torch.Tensor): Temporal coordinates of initial condition points.
        u_i (torch.Tensor): Values of u at initial condition points.
        v_i (torch.Tensor): Values of v at initial condition points.
        p_i (torch.Tensor): Values of p at initial condition points.
        t_b_x (torch.Tensor): Temporal coordinates for left/right boundary points.
        x_b_l (torch.Tensor): Spatial x coordinates for left boundary points.
        x_b_r (torch.Tensor): Spatial x coordinates for right boundary points.
        y_b_x (torch.Tensor): Spatial y coordinates for left/right boundary points.
        t_b_y (torch.Tensor): Temporal coordinates for bottom/top boundary points.
        x_b_y (torch.Tensor): Spatial x coordinates for bottom/top boundary points.
        y_b_b (torch.Tensor): Spatial y coordinates for bottom boundary points.
        y_b_u (torch.Tensor): Spatial y coordinates for top boundary points.
    """

    x_f: torch.Tensor
    y_f: torch.Tensor
    t_f: torch.Tensor
    x_i: torch.Tensor
    y_i: torch.Tensor
    t_i: torch.Tensor
    u_i: torch.Tensor
    v_i: torch.Tensor
    p_i: torch.Tensor
    t_b_x: torch.Tensor
    x_b_l: torch.Tensor
    x_b_r: torch.Tensor
    y_b_x: torch.Tensor
    t_b_y: torch.Tensor
    x_b_y: torch.Tensor
    y_b_b: torch.Tensor
    y_b_u: torch.Tensor


class StokesLoader(BaseLoader):
    """Data loader for generating training tensors for the Navier-Stokes equations.

    Uses the logic extracted from the Stokes' equation notebook.

    Attributes:
        n_f (int): Number of collocation points for the PDE residual.
        n_i (int): Number of initial condition points.
        n_b (int): Number of boundary condition points.
        x_range (tuple[float, float]): Spatial domain x range.
        y_range (tuple[float, float]): Spatial domain y range.
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
        """Initializes the StokesLoader.

        Args:
            n_f (int): Number of collocation points for the PDE residual.
            n_i (int): Number of initial condition points.
            n_b (int): Number of boundary condition points.
            x_range (tuple[float, float]): Spatial domain x range.
            y_range (tuple[float, float]): Spatial domain y range.
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

    def load(self) -> StokesTensors:
        """Generates and returns the StokesTensors container.

        Returns:
            StokesTensors: The generated tensors for the Stokes equation.
        """
        # Collocation points
        x_f = np.random.uniform(self.x_range[0], self.x_range[1], self.n_f)
        y_f = np.random.uniform(self.y_range[0], self.y_range[1], self.n_f)
        t_f = np.random.uniform(self.t_range[0], self.t_range[1], self.n_f)

        # Initial condition points
        x_i = np.random.uniform(self.x_range[0], self.x_range[1], self.n_i)
        y_i = np.random.uniform(self.y_range[0], self.y_range[1], self.n_i)
        t_i = np.zeros_like(x_i)
        u_i = np.sin(x_i) * np.cos(y_i)
        v_i = -np.cos(x_i) * np.sin(y_i)
        p_i = -1 / 4 * (np.cos(2 * x_i) + np.cos(2 * y_i))

        # Boundary condition points
        # Left and right boundary points (x = x_range[0] and x = x_range[1])
        t_b_x = np.random.uniform(self.t_range[0], self.t_range[1], self.n_b)
        x_b_l = np.full_like(t_b_x, self.x_range[0])
        x_b_r = np.full_like(t_b_x, self.x_range[1])
        y_b_x = np.random.uniform(self.y_range[0], self.y_range[1], self.n_b)

        # Bottom and top boundary points (y = y_range[0] and y = y_range[1])
        t_b_y = np.random.uniform(self.t_range[0], self.t_range[1], self.n_b)
        x_b_y = np.random.uniform(self.x_range[0], self.x_range[1], self.n_b)
        y_b_b = np.full_like(t_b_y, self.y_range[0])
        y_b_u = np.full_like(t_b_y, self.y_range[1])

        return StokesTensors(
            x_f=self._to_tensor(x_f, requires_grad=True),
            y_f=self._to_tensor(y_f, requires_grad=True),
            t_f=self._to_tensor(t_f, requires_grad=True),
            x_i=self._to_tensor(x_i),
            y_i=self._to_tensor(y_i),
            t_i=self._to_tensor(t_i),
            u_i=self._to_tensor(u_i),
            v_i=self._to_tensor(v_i),
            p_i=self._to_tensor(p_i),
            t_b_x=self._to_tensor(t_b_x),
            x_b_l=self._to_tensor(x_b_l),
            x_b_r=self._to_tensor(x_b_r),
            y_b_x=self._to_tensor(y_b_x),
            t_b_y=self._to_tensor(t_b_y),
            x_b_y=self._to_tensor(x_b_y),
            y_b_b=self._to_tensor(y_b_b),
            y_b_u=self._to_tensor(y_b_u),
        )
