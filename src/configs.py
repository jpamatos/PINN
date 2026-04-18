from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainerConfig:
    """
    Configuration for the Trainer.

    Attributes:
        _target_: The target class for instantiation.
        _partial_: Whether to partially instantiate the class.
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        weight_ic: Weight for the initial condition loss.
        weight_bc: Weight for the boundary condition loss.
        weight_pde: Weight for the PDE residual loss.
    """

    _target_: str = "training.trainer.Trainer"
    epochs: int = 2000
    lr: float = 1e-3
    weight_ic: float = 50.0
    weight_bc: float = 10.0
    weight_pde: float = 1.0


@dataclass
class BurgersPINNConfig:
    """
    Configuration for the BurgersPINN model.

    Attributes:
        _target_: The target class for instantiation.
        num_hidden_layers: Number of hidden layers in the MLP.
        hidden_units: Number of hidden units per layer.
        nu: Kinematic viscosity parameter for the Burgers equation.
        activation: Activation function to use.
    """

    _target_: str = "pinn.burgers_pinn.BurgersPINN"
    num_hidden_layers: int = 8
    hidden_units: int = 50
    nu: float = 0.01 / torch.pi
    activation: str = "Tanh"


@dataclass
class BurgersDataConfig:
    """
    Configuration for the BurgersData loader.

    Attributes:
        _target_: The target class for instantiation.
        n_f: Number of collocation points for PDE residual.
        n_i: Number of initial condition points.
        n_b: Number of boundary condition points.
        x_range: Spatial domain range.
        t_range: Temporal domain range.
    """

    _target_: str = "data.burgers_data.BurgersLoader"
    n_f: int = 10000
    n_i: int = 200
    n_b: int = 200
    x_range: list[int] = field(default_factory=lambda: [-1, 1])
    t_range: list[int] = field(default_factory=lambda: [0, 1])


@dataclass
class HeatBarPINNConfig:
    """
    Configuration for the HeatBarPINN model.

    Attributes:
        _target_: The target class for instantiation.
        num_hidden_layers: Number of hidden layers in the MLP.
        hidden_units: Number of hidden units per layer.
        activation: Activation function to use.
    """

    _target_: str = "pinn.heat_bar_pinn.HeatBarPINN"
    num_hidden_layers: int = 8
    hidden_units: int = 50
    activation: str = "Tanh"


@dataclass
class HeatBarDataConfig:
    """
    Configuration for the HeatBarData loader.

    Attributes:
        _target_: The target class for instantiation.
        n_f: Number of collocation points for PDE residual.
        n_i: Number of initial condition points.
        n_b: Number of boundary condition points.
        x_range: Spatial x domain range.
        y_range: Spatial y domain range.
        t_range: Temporal domain range.
    """

    _target_: str = "data.heat_bar_data.HeatBarLoader"
    n_f: int = 10000
    n_i: int = 250
    n_b: int = 250
    x_range: list[int] = field(default_factory=lambda: [-1, 1])
    y_range: list[int] = field(default_factory=lambda: [-1, 1])
    t_range: list[int] = field(default_factory=lambda: [0, 1])


@dataclass
class StokesPINNConfig:
    """
    Configuration for the StokesPINN model.

    Attributes:
        _target_: The target class for instantiation.
        num_hidden_layers: Number of hidden layers in the MLP.
        hidden_units: Number of hidden units per layer.
        nu: Kinematic viscosity parameter.
        activation: Activation function to use.
    """

    _target_: str = "pinn.stokes_pinn.StokesPINN"
    num_hidden_layers: int = 8
    hidden_units: int = 50
    nu: float = 0.1
    activation: str = "Tanh"


@dataclass
class StokesDataConfig:
    """
    Configuration for the StokesData loader.

    Attributes:
        _target_: The target class for instantiation.
        n_f: Number of collocation points for PDE residual.
        n_i: Number of initial condition points.
        n_b: Number of boundary condition points.
        x_range: Spatial x domain range.
        y_range: Spatial y domain range.
        t_range: Temporal domain range.
    """

    _target_: str = "data.stokes_data.StokesLoader"
    n_f: int = 10000
    n_i: int = 2000
    n_b: int = 1000
    x_range: list[float] = field(default_factory=lambda: [0.0, 2 * torch.pi])
    y_range: list[float] = field(default_factory=lambda: [0.0, 2 * torch.pi])
    t_range: list[float] = field(default_factory=lambda: [0.0, 1.0])


@dataclass
class Config:
    """
    Main configuration object aggregating all sub-configurations.

    Attributes:
        trainer: Configuration for the trainer.
        model: Configuration for the PINN model.
        data: Configuration for the data loader.
    """

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Any = field(default_factory=StokesPINNConfig)
    data: Any = field(default_factory=StokesDataConfig)
