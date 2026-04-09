import torch


def calculate_grad(
    f: torch.Tensor, x: torch.Tensor, retain_graph: bool = True
) -> torch.Tensor:
    """Computes the gradient of f with respect to x.

    Used for calculating partial derivatives in PDE residuals.

    Args:
        f (torch.Tensor): The function tensor to differentiate.
        x (torch.Tensor): The variable tensor to differentiate with respect to.
        retain_graph (bool, optional): Whether to retain the computational graph. Defaults to True.

    Returns:
        torch.Tensor: The computed gradient tensor.
    """
    return torch.autograd.grad(
        f,
        x,
        grad_outputs=torch.ones_like(f),
        retain_graph=retain_graph,
        create_graph=True,
    )[0]
