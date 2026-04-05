from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseLoader(ABC):
    """
    Base class for PINN data loaders.
    Provides utility methods for tensor conversion and device management.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = device

    def _to_tensor(self, arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        """
        Converts a numpy array to a torch float32 tensor, reshapes it to (-1, 1),
        moves it to the specified device, and sets requires_grad.
        """
        return (
            torch.tensor(arr, dtype=torch.float32)
            .reshape(-1, 1)
            .to(self.device)
            .requires_grad_(requires_grad)
        )

    @abstractmethod
    def load(self) -> Any:
        """
        Abstract method to generate and return the problem-specific tensors.
        """
        pass
