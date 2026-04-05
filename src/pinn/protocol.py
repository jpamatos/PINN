from typing import Any, Protocol

import torch


class PINNProtocol(Protocol):
    """
    Protocol for PINN models.
    Defines the expected interface for forward passes and PDE residual calculations.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Invoke the model's forward pass."""
        ...

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute the model's forward pass."""
        ...

    def pde_residual(
        self, *args: Any, **kwargs: Any
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Compute the PDE residual(s) for the physical system."""
        ...
