import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from data.protocol import DataLoaderProtocol
from pinn.protocol import PINNProtocol
from training.callbacks import MLflowCallback


class Trainer:
    """Trainer for Physics-Informed Neural Networks (PINNs).

    Specifically implements the logic for Burgers equation based on the provided notebook.

    Attributes:
        model (PINNProtocol): The PINN model to train.
        loader (DataLoaderProtocol): Data loader for generating training tensors.
        criterion (nn.Module): Loss function (defaults to nn.MSELoss).
        device (torch.device): The device to run training on (CPU or CUDA).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        weight_ic (float): Weight for the initial condition loss.
        weight_bc (float): Weight for the boundary condition loss.
        weight_pde (float): Weight for the PDE residual loss.
        callbacks (list): List of training callbacks.
    """

    def __init__(
        self,
        model: PINNProtocol,
        loader: DataLoaderProtocol,
        epochs: int,
        lr: float,
        weight_ic: float = 10.0,
        weight_bc: float = 10.0,
        weight_pde: float = 1.0,
    ) -> None:
        """Initializes the Trainer.

        Args:
            model (PINNProtocol): The PINN model to train.
            loader (DataLoaderProtocol): Data loader for generating training tensors.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
            weight_ic (float, optional): Weight for the initial condition loss. Defaults to 10.0.
            weight_bc (float, optional): Weight for the boundary condition loss. Defaults to 10.0.
            weight_pde (float, optional): Weight for the PDE residual loss. Defaults to 1.0.
        """
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.lr = lr
        self.weight_ic = weight_ic
        self.weight_bc = weight_bc
        self.weight_pde = weight_pde
        self.callbacks = []
        mlflow.set_tracking_uri("../mlruns")
        mlflow.set_experiment("pinn")

    def train(self, epochs: int) -> None:
        """Main training loop for the PINN.

        Args:
            epochs (int): Number of epochs to train for.
        """
        if hasattr(self.loader, "device"):
            self.loader.device = self.device
        data = self.loader.load()

        self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-6
        )
        with mlflow.start_run():
            callback = MLflowCallback()
            params = {
                "lr": 1e-3,
                "epochs": epochs,
            }
            params = params | self.model.static_params()
            callback.log_params(params)
            for epoch in range(epochs):
                optimizer.zero_grad()

                loss_ic = self.model.inital_conditions(criterion, data)

                loss_bc = self.model.boundary_conditions(criterion, data)

                loss_pde = self.model.pde_residual(data)

                loss = (
                    self.weight_ic * loss_ic
                    + self.weight_bc * loss_bc
                    + self.weight_pde * loss_pde
                )

                losses = {
                    "loss_ic": loss_ic.item(),
                    "loss_bc": loss_bc.item(),
                    "loss_pde": loss_pde.item(),
                    "loss": loss.item(),
                }
                callback.on_epoch_end(epoch, losses)
                loss.backward()
                optimizer.step()

                scheduler.step(loss.item())

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}/{epochs}: loss={loss.item():.6f}")
