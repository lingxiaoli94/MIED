import torch

def compute_jacobian(outputs, inputs,
                     create_graph=True, retain_graph=True):
    '''
    Compute Jacobian matrices in batch.

    :param outputs: (..., D1)
    :param inputs: (..., D2)
    :returns: (..., D1, D2), computed Jacobian
    '''
    J = torch.cat([
        torch.autograd.grad(
            outputs=outputs[..., d], inputs=inputs,
            create_graph=create_graph, retain_graph=retain_graph,
            grad_outputs=torch.ones(inputs.size()[:-1], device=inputs.device)
        )[0].unsqueeze(-2)
        for d in range(outputs.shape[-1])
    ], -2)  # (..., D1, D2)
    return J

