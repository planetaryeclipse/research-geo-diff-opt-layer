import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from pathlib import Path

import random
from dataclasses import dataclass



def generate_spline(t_final, points = None, dt=0.01, spline_type="cubic"):
    spline_method = CubicSpline  # for convenience
    t = np.linspace(0, t_final, int(t_final / dt))
    if points is None:
        points = np.array([(0.0, 0.0, 0.0), (5.0, 5.0, t_final / 2), (10.0, 0.0, t_final)])


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


def save_data_to_file(filepath, target_trajectory, vehicle_trajectory, controller_output, dt): #saves target x, y | current state x y theta | controller output
    filename = "data"

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    target_trajectory = target_trajectory[:len(vehicle_trajectory), :]
    print(target_trajectory.shape,vehicle_trajectory.shape, controller_output.shape  )

    #Shorten target trajectory isf vehicle trajectory short

    data = np.concatenate([ target_trajectory, vehicle_trajectory, controller_output], axis=1)
    print(data.shape)
    k = 1
    while True:
        filepath = data_dir / f"{filename}{k}.npz"
        if not filepath.exists():
            break
        k += 1

    np.savez_compressed(filepath, data = data)

def _mpc_cost(t, y, u_steps, dt, params, model_step, state_cost):
    # do not consider the current cost
    num_steps = u_steps.shape[0]

    total_cost = 0
    y_upd = y

    for i in range(num_steps):
        u = u_steps[i, :]
        t_upd = t + dt * i

        y_upd = model_step(t_upd, dt, y_upd, u, params)
        total_cost += state_cost(t_upd, y_upd, u, i, params)

    return total_cost

def gen_mpc_controls(
    t, y, steps, dt, u_guess: np.ndarray, params, model_step, state_cost
):
    # must find inputs over finite time horizon that minimize the total cost

    # repeats the guess then flattens to be able to use as variable for minimize
    u_steps_guess = np.repeat(
        u_guess.reshape((1, len(u_guess))), steps, axis=0
    ).flatten()

    result = minimize(
        lambda u_steps: _mpc_cost(
            t,
            y,
            u_steps.reshape((steps, len(u_guess))),  # each step as row
            dt,
            params,
            model_step,
            state_cost,
        ),
        u_steps_guess,
    )

    if not result.success:
        print("Failed to fully converge at solution, is suboptimal")

    result_u_steps = result.x.reshape((steps, len(u_guess)))
    return (result_u_steps[0, :], result_u_steps)

def dyn_ext_unicycle_cost(t, y, u, k, params):
    x, y, theta, v, omega = y
    x_traj, y_traj, theta_traj = params.traj_gen(t)

    state_cost = (
        0.5 * params.dist_cost * ((x_traj - x) ** 2 + (y_traj - y) ** 2)
        + 0.5 * params.ang_cost * (theta - theta_traj) ** 2
        +  (params.neg_v_cost + params.neg_v_cost_slope * abs(v) )* (v < 0)
    )

    input_cost = u.T @ params.u_cost @ u

    return params.fut_pred_factor(k) * state_cost + input_cost

def dyn_ext_unicycle_model_step(t, dt, y, u, params):
    u_f, u_t = u

    def model_f(t, y):
        _, _, theta, v, omega = y
        return np.array([v * np.cos(theta), v * np.sin(theta), omega, u_f, u_t])

    result = solve_ivp(model_f, [0, dt], y)
    return result.y[:, -1]



#Variables
dt = 0.01
t_final = 50.0
num_samples = int(t_final / dt)

#create random points
x_initial_max = 5
y_initial_max = 5
initial_vel_variance = 2
initial_dot_theta_variance = 2
initial_force_variance = 2
initial_torque_variance = 2

x_max = 20
y_max = 20
t1 = random.randint(1, 7)
t2 = random.randint(8, 15)
x0 = random.randint(-x_initial_max, x_initial_max)
y0 = random.randint(-y_initial_max, y_initial_max)
x1 = random.randint(0, x_max-1)
y1 = random.randint(0, y_max)
x2 = random.randint(0, x_max)
y2 = random.randint(0, y_max)
x3 = random.randint(0, x_max)
y3 = random.randint(0, y_max)
theta0 = np.random.uniform(0, np.pi/2)
vel0 = np.random.uniform(-initial_vel_variance, initial_vel_variance)
dot_theta0 = np.random.uniform(-initial_torque_variance, initial_torque_variance)
force0 = np.random.uniform(-initial_dot_theta_variance, initial_dot_theta_variance)
torque0 = np.random.uniform(-initial_torque_variance, initial_torque_variance)

p0 = np.array([x0, y0, theta0, vel0, dot_theta0])  # pointing straight up
u0 = np.array([force0, torque0])



dt = 0.075

num_samples = int(t_final / dt)

spline_method = CubicSpline

t = np.linspace(0, t_final, num_samples)
# path_points = np.array([(0.0, 0.0, 0.0), (x1, y1, t1),(x2, y2, t2), (x3, y3, t_final)])
path_points = np.array([(0.0, 0.0, 0.0), (1.0, 3.0, t_final / 4), (4, 2, 2*t_final/3), (5.0, 0.0, t_final)])


# generators for the spline
x_spline, y_spline = (
    spline_method(path_points[:, 2], path_points[:, 0]),
    spline_method(path_points[:, 2], path_points[:, 1]),
)

# computes the array indices
x_traj, y_traj, x_dot_traj, y_dot_traj = (
    x_spline(t),
    y_spline(t),
    x_spline.derivative()(t),
    y_spline.derivative()(t),
)
(theta_traj, dot_theta_traj) = (
    np.atan2(x_traj, y_traj),
    np.atan2(x_dot_traj, y_dot_traj),
)

# generates a lookup handle
def traj(t):
    closest_idx = int(round(t / dt))
    return (x_traj[closest_idx], y_traj[closest_idx], theta_traj[closest_idx])


@dataclass
class DynExtUnicycleMCPParams:
    dist_cost = 2.5
    ang_cost = 1
    neg_v_cost = 100
    neg_v_cost_slope = 20

    u_cost = 0.5 * np.identity(2)
    traj_gen = traj
    fut_pred_factor = lambda k: 1.0 * k if k > 0 else 1.0




u_mpc = gen_mpc_controls(
    0.0,
    p0,
    10,
    dt,
    np.zeros((2,)),
    DynExtUnicycleMCPParams,
    dyn_ext_unicycle_model_step,
    dyn_ext_unicycle_cost,
)

# implement the control loop

# p0 = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0])  # pointing straight up

p_hist = []
u_hist = []

p_curr = p0
u_curr = u0

num_mpc_steps = 6

for i in range(num_samples):
    t_curr = t[i]

    # generate the next control action
    u_optimal, _ = gen_mpc_controls(
        t_curr,
        p_curr,
        num_mpc_steps,
        0.25,  # use smaller steps
        u_curr,  # uses the previous controls as a guess for the next one
        DynExtUnicycleMCPParams,
        dyn_ext_unicycle_model_step,
        dyn_ext_unicycle_cost,
    )

    # update the histories with the current state and control input before
    # advancing to the next time step
    p_hist.append(p_curr)
    u_hist.append(u_optimal)

    print(f"t: {t_curr}, state: {p_curr}, u_optimal: {u_optimal}")

    # update the model with the optimal control
    p_curr = dyn_ext_unicycle_model_step(
        t_curr, dt, p_curr, u_optimal, DynExtUnicycleMCPParams
    )

    if i > 150:
        break




p_hist = np.array(p_hist)
u_hist = np.array(u_hist)

step = 10
indices = np.arange(0, len(p_hist), step)

plt.subplot(2, 2, 1)
plt.plot(p_hist[:, 0], p_hist[:, 1], label = "actual")
plt.plot(x_traj, y_traj, label = "ideal")

# 2. Add dots that change color over time
# 'c=indices' tells matplotlib to color based on the time step
# 'cmap' defines the color gradient (e.g., 'viridis', 'plasma', 'jet')
scatter_actual = plt.scatter(p_hist[indices, 0], p_hist[indices, 1], 
                             c=indices, cmap='viridis', s=20, label="actual dots", zorder=3)

scatter_ideal = plt.scatter(x_traj[indices], y_traj[indices], 
                            c=indices, cmap='viridis', s=20, marker='x', label="ideal dots", zorder=3)

# Optional: Add a colorbar to show time progression
# plt.colorbar(scatter_actual, label="Time Step")
plt.title("Trajectory")
plt.legend()

# Plot 2
ax1 = plt.subplot(2, 2, 2)
angle_deg = np.degrees(p_hist[:, 2])  # Convert radians to degrees for better visualization

# Plot Angle on the Left Y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Angle (deg)', color=color1)
ax1.plot(angle_deg, color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title("Angle & Velocity")
ax2 = ax1.twinx() 

# Plot Velocity on the Right Y-axis
color2 = 'tab:red'
ax2.set_ylabel('Velocity', color=color2)
ax2.plot(p_hist[:, 3], color=color2, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)


# Plot 3: forces and torques
ax1 = plt.subplot(2, 2, 3)

# Plot Angle on the Left Y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Torque', color=color1)
ax1.plot(u_hist[:, 1], color=color1, linewidth=2) #force
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title("Force and Torque")
ax2 = ax1.twinx() 

# Plot Velocity on the Right Y-axis
color2 = 'tab:red'
ax2.set_ylabel('Force', color=color2)
ax2.plot(u_hist[:, 0], color=color2, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.tight_layout()
plt.show()




print(u_hist[-1])


filepath = ""
# x_curr, y_curr, theta_curr, v_curr, omega_curr = p_hist
# traj_x, traj_y, traj_theta, dot_traj_x, dot_traj_y, dot_traj_theta = traj
traj = np.column_stack((x_traj, y_traj, theta_traj, x_dot_traj, y_dot_traj, dot_theta_traj))
print("traj shape",traj.shape)
save_data_to_file(filepath,traj, p_hist, u_hist, dt)


# # plot the error history of the controller over time
# dist_err_hist = np.sqrt((traj_xs - x_hist[2:])**2 + (traj_ys - y_hist[2:])**2)


# ax.plot(traj_times, dist_err_hist)

# # plt.show()