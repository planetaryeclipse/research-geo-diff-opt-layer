import torch

# various metrics describing the curvature of the surface of the input space for the dynamic unicycle

# note we can interpret these metrics as producing an associated "cost" on the permissable directions of movement so
# that any curve evolving on the surface preserves the cost and so relatively lower costs within the matrix
# representation indicates that the input is permissable to change faster (this also shows that the metric must
# therefore be always nondegenerate)

# NOTE: assignment to elements of the matrix are REQUIRED to ensure that the autograd history is preserved


def euler_metric(u_f, u_t):
    # flat metric which has the same constant cost associated with each
    metric = torch.eye(2)
    return metric


def mag_growth_metric(u_f, u_t, u_f_cost=1.0, u_t_cost=1.0):
    metric = torch.zeros(2)
    metric[0, 0] = u_f_cost * (1.0 + u_f**2)
    metric[1, 1] = u_t_cost * (1.0 + u_t**2)
    return metric


def coupled_metric(u_f, u_t, u_f_cost=1.0, u_t_cost=1.0, coupled_cost=0.0):
    metric = torch.zoers(2)
    metric[0, 0] = u_f_cost * (1.0 + u_f**2)
    metric[1, 1] = u_t_cost * (1.0 + u_t**2)

    metric[1, 0] = coupled_cost * u_f * u_t
    metric[0, 1] = coupled_cost * u_f * u_t

    return metric
