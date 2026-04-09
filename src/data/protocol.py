from typing import Any, Protocol

import torch


class DataLoaderProtocol(Protocol):
    """Protocol for PINN data loaders.

    Defines the expected interface for generating problem-specific tensors.

    Attributes:
        device (str | torch.device): The device on which the data loader operates.
    """

    device: str | torch.device

    def load(self) -> Any:
        """Generates and returns the problem-specific tensors (usually a dataclass).

        Returns:
            Any: A container with the generated tensors.
        """
        ...
