import torch
from mied.utils.batch_jacobian import compute_jacobian

def compute_hessian(func, inputs):
    '''
    Compute Hessianmatrices in batch.

    :param func: (B, D) -> (B,)
    :param inputs: (B, D)
    :returns: (B, D, D)
    '''
    outputs = func(inputs) # (B,)
    grad = compute_jacobian(outputs.unsqueeze(-1), inputs).squeeze(-2) # (B, D)

    result = compute_jacobian(grad, inputs)

    return result

