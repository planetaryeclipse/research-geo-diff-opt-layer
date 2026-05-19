import numpy as np
import torch
from torch.autograd.functional import jacobian
from torch.func import jacfwd, grad, jvp

# inputs = torch.from_numpy(np.array(0.0))

# print(inputs)

# y = torch.sin(inputs)

# j = jacobian(torch.sin, inputs)

# print(y)
# print(j)

x = torch.from_numpy(np.array([0.0, 1.0]))

print(x)


def f(x):
    y = torch.zeros((2,))

    y[0] = torch.sin(x[0]) + x[1]
    y[1] = torch.cos(x[0]) + 2 * x[1]

    return y


print(f(x))
# print(jacobian(f, x, create_graph=True))
print(jacfwd(f)(x))  # this is the functionality I want

e1 = torch.zeros_like(x)
e1[0] = 1.0

e2 = torch.zeros_like(x)
e2[1] = 1.0

print(jvp(f, (x,), (e1,)))

print(jacfwd(f))

# print(jacobian(jacobian(f, x, create_graph=True, vectorize=True), x))

# ----------------------------

# x = 0.32 * torch.ones(1)
# print(x)


# def f(x):
#     y = torch.zeros((2,))

#     y[0] = torch.sin(x)
#     y[1] = torch.cos(x) + x

#     return y

# print(jacobian(f, x, create_graph=True))
