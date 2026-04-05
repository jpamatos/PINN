from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BasePINN(nn.Module, ABC):
    """
    Base class for Physics-Informed Neural Networks (PINNs).
    Provides a configurable MLP backbone and defines the interface for PDE residuals.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module | None = None,
        num_hidden_layers: int = 8,
        hidden_units: int = 50,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.Tanh()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(activation)

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)

        # Output layer
        layers.append(nn.Linear(hidden_units, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        By default, it assumes x is a single tensor containing all input coordinates.
        """
        return self.net(x)

    @abstractmethod
    def pde_residual(self, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Compute the PDE residual(s).
        Must be implemented by subclasses to define the specific physics.
        """
        pass
