from typing import Any, Protocol

import torch


class PINNProtocol(Protocol):
    """Protocol for PINN models.

    Defines the expected interface for forward passes and PDE residual calculations.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Invokes the model's forward pass.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Model output.
        """
        ...

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Computes the model's forward pass.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Model output.
        """
        ...

    def pde_residual(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Computes the PDE residual(s) for the physical system.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: The computed PDE residual(s).
        """
        ...

    def to(self, device: str | torch.device) -> "PINNProtocol":
        """Moves the model to the specified device.

        Args:
            device (str | torch.device): The device to move the model to.

        Returns:
            PINNProtocol: The model instance itself.
        """
        ...

    def inital_conditions(self, criterion: Any, data: Any) -> torch.Tensor:
        """Computes the initial conditions loss.

        Args:
            criterion (Any): Loss function.
            data (Any): Container with initial condition data.

        Returns:
            torch.Tensor: Initial condition loss.
        """
        ...

    def boundary_conditions(self, criterion: Any, data: Any) -> torch.Tensor:
        """Computes the boundary conditions loss.

        Args:
            criterion (Any): Loss function.
            data (Any): Container with boundary condition data.

        Returns:
            torch.Tensor: Boundary condition loss.
        """
        ...

    def parameters(self) -> torch.Tensor:
        """Returns the model's parameters.

        Returns:
            dict[str, Any]: Model parameters.
        """
        ...

    def static_params(self) -> dict[str, Any]:
        """Returns the model's static parameters.

        Returns:
            dict[str, Any]: A dictionary of the model's static parameters.
        """
        ...
