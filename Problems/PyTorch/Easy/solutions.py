import torch


# Problem 884
def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    x = torch.Tensor([float(x_value)]).requires_grad_()

    # Pass through func
    y = x.pow(2) + x.mul(3) + 3

    # Compute gradient
    y.backward()

    return x.grad.item()
