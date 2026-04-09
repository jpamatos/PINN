from typing import Any

import mlflow


class MLflowCallback:
    """Callback for logging training metrics and parameters to MLflow."""

    def __init__(self, experiment_name: str | None = None) -> None:
        """Initializes the MLflowCallback.

        Args:
            experiment_name (str | None, optional): Name of the MLflow experiment.
                If provided, sets the active experiment. Defaults to None.

        Raises:
            ImportError: If the mlflow package is not installed.
        """
        if mlflow is None:
            raise ImportError(
                "mlflow is not installed. Please install it to use MLflowCallback."
            )
        if experiment_name:
            mlflow.set_experiment(experiment_name)

    def on_epoch_end(self, epoch: int, losses_dict: dict[str, float]) -> None:
        """Logs a dictionary of metrics to MLflow at the end of an epoch.

        Args:
            epoch (int): The current epoch number.
            losses_dict (dict[str, float]): Dictionary containing metric names and values.
        """
        for name, value in losses_dict.items():
            mlflow.log_metric(name, value, step=epoch)

    def log_params(self, params_dict: dict[str, Any]) -> None:
        """Logs a dictionary of parameters to MLflow.

        Args:
            params_dict (dict[str, Any]): Dictionary of parameter names and their values.
        """
        for name, value in params_dict.items():
            mlflow.log_param(name, value)
