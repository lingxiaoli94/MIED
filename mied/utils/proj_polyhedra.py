import torch


def proj_halfspace(p, c, y):
    '''
    Project p to halfspace defined by {x: c^T x <= y}.
    :param p: (B, D)
    :param c: (B, D)
    :param y: (B,)
    :return: (B, D), projected points
    '''

    norm = torch.norm(c, dim=-1) + 1e-8 # (B,)
    c = c / norm.unsqueeze(-1) # (B, D)
    y = y / norm # (B,)

    dot = (p * c).sum(-1) # (B,)
    return p - (dot - y).relu().unsqueeze(-1) * c


def calc_suboptimality(X, C, Y):
    '''
    Calculate the suboptimality for projecting X onto the polyhedral
    defined by C and Y.
    :param X: (B, D)
    :param C: (B, K, D)
    :param Y: (B, K)
    :return: scalar, representing average suboptimality
    '''
    return (torch.matmul(C, X.unsqueeze(-1)).squeeze(-1) - Y).relu().mean()


def proj_polyhedra(X, C, Y,
                  parallel=False,
                  max_num_itr=50, logging=False, early_stop_eps=1e-6):
    '''
    Project each X to the intersection of {C_i^T x <= Y_i, for all i < K}.
    :param X: (B, D)
    :param C: (B, K, D)
    :param Y: (B, K)
    :return: (B, D), projected points
    '''

    if logging:
        hist_loss = [calc_suboptimality(X, C, Y)]

    if C.shape[1] == 1:
        # Single constraint.
        sol = proj_halfspace(X, C[:, 0, :], Y[:, 0])
    else:
        with torch.no_grad():
            K = C.shape[1]
            D = C.shape[2]
            if parallel:
                u_prev_stack = X.unsqueeze(1).expand(-1, K, -1) # (B, K, D)
                z_prev_stack = torch.zeros_like(u_prev_stack) # (B, K, D)
            else:
                u_prev_list = []
                z_prev_list = []
                for _ in range(K + 1):
                    u_prev_list.append(X.clone().detach())
                    z_prev_list.append(torch.zeros_like(X))

            for _ in range(max_num_itr):
                if parallel:
                    u0 = u_prev_stack.mean(1) # (B, D)
                    tmp = u0.unsqueeze(1) + z_prev_stack # (B, K, D)
                    u_next_stack = proj_halfspace(tmp.reshape(-1, D),
                                                  C.reshape(-1, D),
                                                  Y.reshape(-1)).reshape(-1, K, D) # (B, K, D)
                    z_next_stack = tmp - u_next_stack

                    u_prev_stack = u_next_stack
                    z_prev_stack = z_next_stack
                else:
                    u_next_list = []
                    u_next_list.append(u_prev_list[K])
                    z_next_list = [None]
                    for i in range(K):
                        tmp = u_next_list[i] + z_prev_list[i + 1]
                        u_next_list.append(proj_halfspace(tmp,
                                                          C[:, i, :], Y[:, i]))
                        z_next_list.append(tmp - u_next_list[-1])
                    u_prev_list = u_next_list
                    z_prev_list = z_next_list

                sol = u_prev_stack.mean(1) if parallel else u_prev_list[-1]
                if logging:
                    subopt = calc_suboptimality(
                        sol,
                        C, Y)
                    hist_loss.append(subopt)
                    if subopt < early_stop_eps:
                        break

    if logging:
        return sol, hist_loss
    return sol
