from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, UnivariateSpline
import torch
from torch._numpy import float64
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.module import Module
from pathlib import Path
from optimal_traj_control.automated_test.optimal_control import (
    dynamic_unicycle_optimal_traj_derivations,
    convert_sym_prob_to_cvxpy_prob,
)
import random


def generate_spline(t_f, points = None, dt=0.01, spline_type="cubic"):
    spline_method = CubicSpline  # for convenience
    t = np.linspace(0, t_f, int(t_f / dt))
    if points is None:
        points = np.array(
            # defines a smoothed triangular path
            [
                # (x, y, t)
                (0.0, 0.0, 0.0),
                (5.0, 5.0, 15.0),
                (10.0, 0.0, 20.0),
            ]
        )

    (x_spline, y_spline) = (
        spline_method(points[:, 2], points[:, 0]),
        spline_method(points[:, 2], points[:, 1]),
    )
    (traj_x, traj_y, dot_traj_x, dot_traj_y) = (
        x_spline(t),
        y_spline(t),
        x_spline.derivative()(t),
        y_spline.derivative()(t),
    )

    (traj_theta, dot_traj_theta) = (
        np.atan2(traj_y, traj_x),
        np.atan2(dot_traj_y, dot_traj_x),
    )
    traj = (traj_x, traj_y, traj_theta, dot_traj_x, dot_traj_y, dot_traj_theta)
    # fig, ax = plt.subplots()
    # ax.plot(traj_x, traj_y)

    # plt.show()
    print("traj_x" ,len(traj_x))
    return traj, t


def save_data_to_file(filepath,t, target_trajectory, vehicle_trajectory, controller_output, dt): #saves target x, y | current state x y theta | controller output
    filename = "data"

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)


    t = t.reshape(-1, 1)
    print(t.shape,target_trajectory.shape,vehicle_trajectory.shape, controller_output.shape  )

    data = np.concatenate([t, target_trajectory, vehicle_trajectory, controller_output], axis=1)
    print(data.shape)
    k = 1
    while True:
        filepath = data_dir / f"{filename}{k}.npz"
        if not filepath.exists():
            break
        k += 1

    np.savez_compressed(filepath, data = data)


#Variables
dt = 0.01
t_f = 5

#create random points
x_max = 20
y_max = 20
t1 = random.randint(1, 18)
t2 = random.randint(t1, 19)
x1 = random.randint(0, x_max)
y1 = random.randint(0, y_max)
x2 = random.randint(0, x_max)
y2 = random.randint(0, y_max)
x3 = random.randint(0, x_max)
y3 = random.randint(0, y_max)

points = np.array(
    # defines a smoothed triangular path
    [
        # (x, y, t)
        (0.0, 0.0, 0.0),
        (x1, y1, t1),
        (x2, y2, t2),
        (x3, y3, t_f),
    ]
)

(sympy_f, sympy_gs, sympy_opt_vars) = dynamic_unicycle_optimal_traj_derivations()
(cntrl_prob_cvxpy, prob_var_map, prob_param_map) = convert_sym_prob_to_cvxpy_prob(
    sympy_f, sympy_gs, sympy_opt_vars
)


def get_param_var(mapping, idxs):
    params = []
    for idx in idxs:
        pair = mapping[idx]

        print(f"\tSympy: {pair[0]}")
        params.append(pair[1])
    return tuple(params)

# restores handles to optimization variables (cvxpy)
print("Variables:")
(u_f_var, u_t_var, delta_var) = get_param_var(prob_var_map, (0, 1, 2))


# restores handles to parameters (cvxpy)
print("Parameters:")
p_par = get_param_var(prob_param_map, (0,))[0]
(alpha_par, beta_par) = get_param_var(
    prob_param_map, (1, 8)
)
(x_par, y_par, theta_par, v_par, omega_par) = get_param_var(
    prob_param_map, (3, 6, 14, 10, 9)
)

(x_traj_par, y_traj_par, theta_traj_par) = get_param_var(
    prob_param_map, (2, 5, 13, )
)
(
    dot_x_traj_par,
    dot_y_traj_par,
    dot_theta_traj_par,
) = get_param_var(prob_param_map, (11, 12, 15))

# as cvxpy is a convex optimization library we must use these auxiliarly
# variables as replacement for sin(theta) and cos(theta)
# NOTE: given theta is a parameter set at each controller update step we can do
# this without loss of convexity with respect to optimization variables u_f, u_t
(sin_theta_aux_par, cos_theta_aux_par) = get_param_var(prob_param_map, (7, 4))

# set the relative importance for each error

alpha_par.value = 1.0  # distance loss
beta_par.value = 1.0  # angle loss

p_par.value = 0.2

# gamma_par.value = 1e-3  # IGNORE: velocity loss
# delta_par.value = 1e-3  # IGNORE: angular velocity loss

traj, t  = generate_spline(t_f,
     dt=dt, spline_type="cubic")

traj_x, traj_y, traj_theta, dot_traj_x, dot_traj_y, dot_traj_theta = traj

p0 = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0])  # pointing upwards
u0 = np.array([0.0, 0.0])


print([param.value for param in cntrl_prob_cvxpy.parameters()])
print(cntrl_prob_cvxpy.parameters()[0] is alpha_par)


def dyn_unicycle_f(t, p, u):
    x, y, theta, v, omega = p
    u_f, u_t = u

    dot_state = np.array([v * np.cos(theta), v * np.sin(theta), omega, u_f, u_t])
    return dot_state


p_hist = []
u_hist = []

p = p0
u = u0

for i in range(len(t)):
    # generate controller input form the optimal controller

    p_hist.append(p)
    u_hist.append(u)

    # gets the current state
    x_curr, y_curr, theta_curr, v_curr, omega_curr = p

    # gets the current trajectory state
    traj_x_curr, traj_y_curr, traj_theta_curr = (traj_x[i], traj_y[i], traj_theta[i])
    print(f"Traj x: {traj_x_curr}, traj y: {traj_y}")
    dot_traj_x_curr, dot_traj_y_curr, dot_traj_theta_curr = (
        dot_traj_x[i],
        dot_traj_y[i],
        dot_traj_theta[i],
    )

    # set parameters of the optimization problem

    x_par.value = x_curr
    y_par.value = y_curr
    theta_par.value = theta_curr
    v_par.value = v_curr
    omega_par.value = omega_curr

    x_traj_par.value = traj_x_curr
    y_traj_par.value = traj_y_curr
    theta_traj_par.value = traj_theta_curr
    # v_traj_par.value = 0.0  # not generated and error gain disabled
    # omega_traj_par.value = 0.0  # not generated and error gain disabled

    dot_x_traj_par.value = dot_traj_x_curr
    dot_y_traj_par.value = dot_traj_y_curr
    dot_theta_traj_par.value = dot_traj_theta_curr
    # dot_v_traj_par.value = 0.0  # not generated and error gain disabled
    # dot_omega_traj_par.value = 0.0  # not generated and error gain disabled

    print([param.value for param in cntrl_prob_cvxpy.parameters()])

    # set the auxiliary variables
    sin_theta_aux_par.value = np.sin(theta_curr)
    cos_theta_aux_par.value = np.cos(theta_curr)

    # run optimization

    cntrl_prob_cvxpy.solve(warm_start=True)
    print(cntrl_prob_cvxpy.constraints[0].value())

    u_f_optim = u_f_var.value
    u_t_optim = u_t_var.value
    delta_optim = delta_var.value

    print(f"u_f: {u_f_optim}, u_t: {u_t_optim}, delta: {delta_optim}")

    u = np.array([u_f_optim, u_t_optim])

    # update history

    # run dynamics (zero order hold on dynamics for dt)
    result = solve_ivp(lambda t, y: dyn_unicycle_f(t, y, u=u), [0, dt], p)
    print(result.y)

    p = result.y[:, -1]
    # x_curr, y_curr, theta_curr, v_curr, omega_curr = p

    if i > 2000:
        break

p_hist = np.array(p_hist)
u_hist = np.array(u_hist)

fig, ax = plt.subplots()

ax.plot(p_hist[:, 0], p_hist[:, 1])
ax.plot(traj_x, traj_y)




plt.show()

print(u_hist[-1])


filepath = ""
# x_curr, y_curr, theta_curr, v_curr, omega_curr = p_hist
traj_x, traj_y, traj_theta, dot_traj_x, dot_traj_y, dot_traj_theta = traj
traj = np.column_stack((traj_x, traj_y, traj_theta, dot_traj_x, dot_traj_y, dot_traj_theta))

# save_data_to_file(filepath,t ,traj, p_hist, u_hist, dt)


# # plot the error history of the controller over time
# dist_err_hist = np.sqrt((traj_xs - x_hist[2:])**2 + (traj_ys - y_hist[2:])**2)


# ax.plot(traj_times, dist_err_hist)

# # plt.show()