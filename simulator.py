from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, UnivariateSpline




def generate_spline(points, dt=0.01, spline_type="cubic", v=1.0):
    """
    Generate a parametric spline from 2D points and sample at constant velocity.

    Parameters:
    -----------
    points : list[tuple[float, float]] | np.ndarray
        Sequence of (x, y) points.
    dt : float
        Time step for sampling (default 0.01)
    spline_type : str
        'cubic' for CubicSpline or 'univariate' for UnivariateSpline
    v : float
        Constant velocity for arc length parameterization (default 1.0)

    Returns:
    --------
    spline_x, spline_y : spline functions
        Parametric splines x(t) and y(t) where t is parameter from 0 to 1
    traj : np.ndarray
        Array of (x, y, v) points spaced at constant velocity intervals
        where v is the velocity at each point
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a sequence of (x, y) pairs")

    n_points = len(pts)

    # Create parameter t from 0 to 1
    t_param = np.linspace(0, 1, n_points)

    # Create parametric splines: x(t) and y(t)
    if spline_type == "cubic":
        spline_x = CubicSpline(t_param, pts[:, 0])
        spline_y = CubicSpline(t_param, pts[:, 1])
    elif spline_type == "univariate":
        spline_x = UnivariateSpline(t_param, pts[:, 0], s=0)
        spline_y = UnivariateSpline(t_param, pts[:, 1], s=0)
    else:
        raise ValueError("spline_type must be 'cubic' or 'univariate'")

    # Calculate arc length function s(t) by integrating |r'(t)| dt
    # First, get a fine sampling of the curve
    t_fine = np.linspace(0, 1, 1000)
    x_fine = spline_x(t_fine)
    y_fine = spline_y(t_fine)

    # Calculate derivatives
    dx_dt = spline_x.derivative()(t_fine)
    dy_dt = spline_y.derivative()(t_fine)

    # Arc length element: ds = sqrt(dx^2 + dy^2) * dt
    ds_dt = np.sqrt(dx_dt**2 + dy_dt**2)

    # Cumulative arc length (integrate ds/dt)
    dt_fine = t_fine[1] - t_fine[0]  # uniform spacing
    arc_lengths = np.zeros_like(t_fine)
    arc_lengths[1:] = np.cumsum(ds_dt[:-1]) * dt_fine
    total_length = arc_lengths[-1]

    # Create interpolation: t(s) where s is arc length
    from scipy.interpolate import interp1d

    t_of_s = interp1d(
        arc_lengths, t_fine, kind="linear", bounds_error=False, fill_value=(0, 1)
    )

    # Sample at constant velocity: s = v * t, where t is time
    # Total time = total_length / v
    total_time = total_length / v
    time_points = np.arange(0, total_time + dt, dt)

    # Convert time to arc length
    arc_length_points = v * time_points

    # Clamp to valid range
    arc_length_points = np.clip(arc_length_points, 0, total_length)

    # Convert arc length to parameter t
    t_points = t_of_s(arc_length_points)

    # Evaluate splines at these parameter values
    xs_targ = spline_x(t_points)
    ys_targ = spline_y(t_points)

    # Create velocity array (constant velocity v for all points)
    vs_targ = np.full_like(xs_targ, v)

    # Stack x, y, and velocity into trajectory array
    traj = np.column_stack((xs_targ, ys_targ, vs_targ))

    return (spline_x, spline_y), traj


def plot_both_trajectories_with_time(
    ideal_xs,
    ideal_ys,
    ideal_times,
    ode_xs,
    ode_ys,
    ode_times,
    ax=None,
    figsize=(12, 10),
    ideal_colormap="viridis",
    ode_colormap="viridis",
    ideal_label="Ideal Trajectory",
    ode_label="ODE Trajectory",
):
    """
    Plot both ideal and ODE trajectories on the same plot with time visualization.

    Parameters:
    -----------
    ideal_xs : array-like
        X positions of ideal trajectory
    ideal_ys : array-like
        Y positions of ideal trajectory
    ideal_times : array-like
        Time values for ideal trajectory
    ode_xs : array-like
        X positions of ODE trajectory (handles nested lists)
    ode_ys : array-like
        Y positions of ODE trajectory (handles nested lists)
    ode_times : array-like
        Time values for ODE trajectory
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    figsize : tuple
        Figure size (width, height)
    ideal_colormap : str
        Colormap for ideal trajectory (default: 'viridis')
    ode_colormap : str
        Colormap for ODE trajectory (default: 'viridis')
    ideal_label : str
        Label for ideal trajectory
    ode_label : str
        Label for ODE trajectory

    Returns:
    --------
    fig : matplotlib figure
        The figure object
    ax : matplotlib axes
        The axes object
    """

    def flatten_to_array(data):
        """Convert nested lists/arrays to flat 1D array."""
        result = []
        for item in data:
            if isinstance(item, (list, tuple, np.ndarray)):
                if len(item) > 0:
                    if isinstance(item[0], (list, tuple, np.ndarray)):
                        result.append(float(item[0][0]) if len(item[0]) > 0 else 0.0)
                    else:
                        result.append(float(item[0]))
                else:
                    result.append(0.0)
            else:
                result.append(float(item))
        return np.array(result, dtype=float)

    # Flatten ODE trajectory arrays
    ode_xs = flatten_to_array(ode_xs)
    ode_ys = flatten_to_array(ode_ys)

    # Convert to numpy arrays
    ideal_xs = np.asarray(ideal_xs, dtype=float)
    ideal_ys = np.asarray(ideal_ys, dtype=float)
    ideal_times = np.asarray(ideal_times, dtype=float)
    ode_times = np.asarray(ode_times, dtype=float)

    # Ensure same length for ODE trajectory
    min_len = min(len(ode_xs), len(ode_ys), len(ode_times))
    ode_xs = ode_xs[:min_len]
    ode_ys = ode_ys[:min_len]
    ode_times = ode_times[:min_len]

    # Find common time range for normalization
    time_min = min(ideal_times.min(), ode_times.min())
    time_max = max(ideal_times.max(), ode_times.max())
    norm_combined = plt.Normalize(vmin=time_min, vmax=time_max)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot ideal trajectory with time coloring
    from matplotlib.collections import LineCollection

    points_ideal = np.array([ideal_xs, ideal_ys]).T.reshape(-1, 1, 2)
    segments_ideal = np.concatenate([points_ideal[:-1], points_ideal[1:]], axis=1)
    cmap_ideal = plt.get_cmap(ideal_colormap)
    lc_ideal = LineCollection(
        segments_ideal,
        cmap=cmap_ideal,
        norm=norm_combined,
        linewidth=2.5,
        alpha=0.7,
        label=ideal_label,
    )
    lc_ideal.set_array(ideal_times)
    ax.add_collection(lc_ideal)

    # Plot ODE trajectory with time coloring
    points_ode = np.array([ode_xs, ode_ys]).T.reshape(-1, 1, 2)
    segments_ode = np.concatenate([points_ode[:-1], points_ode[1:]], axis=1)
    cmap_ode = plt.get_cmap(ode_colormap)
    lc_ode = LineCollection(
        segments_ode,
        cmap=cmap_ode,
        norm=norm_combined,
        linewidth=2,
        alpha=0.8,
        linestyle="--",
        label=ode_label,
    )
    lc_ode.set_array(ode_times)
    ax.add_collection(lc_ode)

    # Add colorbar (using one of the collections for the scale)
    cbar = plt.colorbar(lc_ideal, ax=ax)
    cbar.set_label("Time", rotation=270, labelpad=15)

    # Format axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.grid(True, alpha=0.3)
    ax.set_title("Ideal vs ODE Trajectory (colored by time)")
    ax.legend()
    plt.tight_layout()

    return fig, ax

# fun = make_fun(v_fn, omega_fn)
class solver:
    def __init__(
        self, m=1, r=0.01, dt=0.01, state0=None
    ):  # change so you can pass in init pos and init vel
        # state0: [x_0, y_0, theta_0, vel_0, theta_dot_0]

        if state0 is None:
            state0 = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.m = m
        self.r = r
        self.dt = dt
        self.xs, self.ys, self.thetas = [[v] for v in state0[:3]]
        self.v_dots, self.theta_dots = [[v] for v in state0[3:]]
        # Initialize x_dots and y_dots as lists (not numpy arrays) so we can append later
        self.x_dots = [self.v_dots[0] * np.cos(self.thetas[0])]
        self.y_dots = [self.v_dots[0] * np.sin(self.thetas[0])]

        # gives second position values for second derivative solver by using velocity
        self.xs.insert(
            0, (self.xs[0] - self.v_dots[0] * np.cos(self.thetas[-1]) * self.dt)
        )
        self.ys.insert(
            0, (self.ys[0] - self.v_dots[0] * np.sin(self.thetas[-1]) * self.dt)
        )
        self.thetas.insert(0, (self.thetas[0] - self.theta_dots[0] * self.dt))

        # self.x_dots.insert(0, 0)
        # self.y_dots.insert(0, 0)
        # self.theta_dots.insert(0, 0)

        print(self.xs, self.ys, self.thetas)

    def solve(
        self, f, f_theta, torque
    ) -> None:  # advances state by time value, returns states
        # position solves

        # force is a vector. change how you can apply force in different directions
        self.xs.append(
            (f / self.m * np.cos(self.thetas[-1]) * self.dt * 2)
            + 2 * self.xs[-1]
            - self.xs[-2]
        )
        self.ys.append(
            (f / self.m * np.sin(self.thetas[-1]) * self.dt * 2)
            + 2 * self.ys[-1]
            - self.ys[-2]
        )
        self.thetas.append(
            (2 / self.m / self.r**2 * torque * self.dt**2)
            + 2 * self.thetas[-1]
            - self.thetas[-2]
        )

        # #vel solves
        # self.x_dots.append(f/self.m * np.cos(self.thetas[-1])*self.dt + self.x_dots[-1])
        # self.y_dots.append(f/self.m * np.sin(self.thetas[-1])*self.dt + self.y_dots[-1])
        # self.theta_dots.append(2 / self.m / self.r**2* torque *self.dt + self.theta_dots[-1])

        # print(len(self.xs), len(self.ys), len(self.x_dots), len(self.theta_dots)) # , self.x_dots, self.y_dots, self.theta_dots

    @property
    def history(self):
        return (self.xs, self.ys, self.thetas)

    @property
    def state(self):
        return (self.xs[-1], self.ys[-1], self.thetas[-1])


class controller:
    def __init__(self, traj, kd_lin=0, kp_lin=0, kd_ang=0, kp_ang=0, dt=0.01):
        if traj is None:
            traj = [[0.0, 0.0]]
        self.traj = traj
        self.dt = dt
        self.kd_lin = kd_lin
        self.kp_lin = kp_lin
        self.kd_ang = kd_ang
        self.kp_ang = kp_ang

        # controller error history so we can evaluate derivatives
        self.prev_lin_err = 0.0
        self.prev_ang_err = 0.0

    def PD_control(
        self, traj_target, state
    ):  # Targ = [x, y, vel] state = [x, y, theta]       , x_dot, y_dot, theta_dot]

        x_traj, y_traj, _ = traj_target
        x, y, theta = state

        # Position errors
        dx = x_traj - x
        dy = y_traj - y
        err_lin = np.sqrt(dx**2 + dy**2)

        # rotates the dx, dy into the local frame of the robot to minimize the error
        local_frame = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        local_dist_err = local_frame.T @ np.array([dx, dy])
        local_dx, local_dy = (local_dist_err[0], local_dist_err[1])
        err_ang = np.atan2(local_dy, local_dx)

        # computes the approximated derivatives (not filtered)
        dot_err_ang = (err_ang - self.prev_ang_err) / self.dt
        dot_err_lin = (err_lin - self.prev_lin_err) / self.dt

        # computes controller inputs
        f = self.kp_lin * err_lin + self.kd_lin * dot_err_lin
        torque = self.kp_ang * err_ang + self.kd_ang * dot_err_ang

        # cache the error history
        self.prev_lin_err = err_lin
        self.prev_ang_err = err_ang

        return f, torque


"""MAIN SCODE BEGINS HERE"""
# initialize starting variables
t_0, t_f, t, dt = 0, 10, 0, 0.01
m, r = 1, 0.1
control_points = [(0, 0), (1, 4), (9, 7), (10, 10)]
x_max, y_max = 20.0, 20.0  # graph x and y axis max
v_traj = 1
# Old force and torque functions
f_fn = lambda t: -1.0 * np.sin(2 * t)
alpha_fn = lambda t: -1 * np.sin(t) + 0.01 * t

f, torque = 0, 0
# initial state
state0 = [0, 0, 1.7, 0, 0]  # should give in x,y,theta, vel, ang_vel
(spline_x, spline_y), traj = generate_spline(
    control_points, dt=dt, spline_type="cubic", v=v_traj
)

solver = solver(m, r, dt, state0)
controller = controller(traj, kd_lin=0.5, kp_lin=0.5, kd_ang=0.5, kp_ang=1.0, dt=dt)

# solver loop - track time for visualization
times = []
for i in range(len(traj)):
    # f = f_fn(t)
    # torque = alpha_fn(t)
    solver.solve(f, 0, torque)  # force, force_angle, torque

    current_traj = traj[i, :]
    f, torque = controller.PD_control(current_traj, solver.state)

    times.append(t)
    t += dt

# debugging
# print_results(xs, ys, thetas, traj=traj, times=times, dt=dt)


# Create spline plot (returns fig, ax)
pts = np.array(control_points)
# fig, ax = visualize_spline_2d((spline_x, spline_y), pts[:, 0], pts[:, 1], x_max, y_max)

# Plot trajectory on the SAME axes (regular plot)
# plot_xy_trajectory(xs, ys, ax=ax, label="ODE Trajectory", linestyle="--")

# ax.legend()
# plt.show()

# Create a combined plot with time visualization for both trajectories
# Extract ideal trajectory (spline) points

traj_xs = traj[:, 0]
traj_ys = traj[:, 1]
traj_times = np.arange(0, len(traj_xs) * dt, dt)[: len(traj_xs)]

# Extract ODE trajectory points and times
times_array = np.array(times, dtype=float)

(x_hist, y_hist, _) = solver.history

# Plot both trajectories with time visualization
fig2, ax2 = plot_both_trajectories_with_time(
    ideal_xs=traj_xs,
    ideal_ys=traj_ys,
    ideal_times=traj_times,
    ode_xs=x_hist,
    ode_ys=y_hist,
    ode_times=times_array,
    ode_colormap="rainbow",
)
plt.show()

# plot the error history of the controller over time
dist_err_hist = np.sqrt((traj_xs - x_hist[2:])**2 + (traj_ys - y_hist[2:])**2)

fig, ax = plt.subplots()
ax.plot(traj_times, dist_err_hist)

plt.show()