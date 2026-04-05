from typing import Any, Protocol

import torch


class DataLoaderProtocol(Protocol):
    """
    Protocol for PINN data loaders.
    Defines the expected interface for generating problem-specific tensors.
    """

    device: str | torch.device

    def load(self) -> Any:
        """
        Generate and return the problem-specific tensors (usually a dataclass).
        """
        ...
