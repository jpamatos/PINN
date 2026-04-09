from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BasePINN(nn.Module, ABC):
    """Base class for Physics-Informed Neural Networks (PINNs).

    Provides a configurable MLP backbone and defines the interface for PDE residuals.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str,
        num_hidden_layers: int = 8,
        hidden_units: int = 50,
    ) -> None:
        """Initializes the BasePINN.

        Args:
            input_dim (int): The dimensionality of the input.
            output_dim (int): The dimensionality of the output.
            activation (str): The name of the activation function to use (from torch.nn).
            num_hidden_layers (int, optional): Number of hidden layers in the MLP. Defaults to 8.
            hidden_units (int, optional): Number of units per hidden layer. Defaults to 50.
        """
        super().__init__()
        activation_fn = getattr(nn, activation)
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(activation_fn())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation_fn())

        # Output layer
        layers.append(nn.Linear(hidden_units, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.

        By default, it assumes x is a single tensor containing all input coordinates.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the neural network.
        """
        return self.net(x)

    @abstractmethod
    def pde_residual(self, *args, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Computes the PDE residual(s).

        Must be implemented by subclasses to define the specific physics.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: The computed PDE residual(s).
        """
        pass
