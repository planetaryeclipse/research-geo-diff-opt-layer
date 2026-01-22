from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, UnivariateSpline

#use rk4 to solve motion ODE
x_dot, y_dot, v, theta, omega = 0, 0, 0, 0, 0

x_dot = v*np.cos(theta)
y_dot = v* np.sin(theta)




def generate_spline(points, spline_type="cubic"):
    """
    Generate a spline from explicitly provided 2D points.

    Parameters:
    -----------
    points : list[tuple[float, float]] | np.ndarray
        Sequence of (x, y) points.
    spline_type : str
        'cubic' for CubicSpline or 'univariate' for UnivariateSpline

    Returns:
    --------
    spline, x_points, y_points
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a sequence of (x, y) pairs")

    x_points = pts[:, 0]
    y_points = pts[:, 1]

    # Sort by x (required for splines)
    sort_idx = np.argsort(x_points)
    x_points = x_points[sort_idx]
    y_points = y_points[sort_idx]

    # Ensure x is strictly increasing (CubicSpline requires this)
    if np.any(np.diff(x_points) <= 0):
        raise ValueError("x values must be strictly increasing (no duplicates)")

    if spline_type == "cubic":
        spline = CubicSpline(x_points, y_points)
    elif spline_type == "univariate":
        spline = UnivariateSpline(x_points, y_points, s=0)
    else:
        raise ValueError("spline_type must be 'cubic' or 'univariate'")

    return spline, x_points, y_points


def visualize_spline_2d(spline, x_points, y_points, x_max, y_max, 
                         n_eval_points=200, figsize=(10, 6), show_points=True):
    """
    Visualize a 2D spline curve with matplotlib.
    
    Parameters:
    -----------
    spline : scipy interpolation object
        The spline function to visualize
    x_points : array
        X coordinates of the control points
    y_points : array
        Y coordinates of the control points
    x_max : float
        Maximum x value for evaluation range
    y_max : float
        Maximum y value (for axis limits)
    n_eval_points : int
        Number of points to evaluate for smooth curve
    figsize : tuple
        Figure size (width, height)
    show_points : bool
        Whether to show the control points
    
    Returns:
    --------
    fig : matplotlib figure
        The figure object
    ax : matplotlib axes
        The axes object
    """
    # Evaluate spline at many points for smooth plotting
    x_eval = np.linspace(x_points.min(), x_points.max(), n_eval_points)
    y_eval = spline(x_eval)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the spline curve
    ax.plot(x_eval, y_eval, 'b-', label='Spline Curve', linewidth=2.5, zorder=1)
    
    # Plot control points if requested
    if show_points:
        ax.plot(x_points, y_points, 'ro', markersize=10, 
                label='Control Points', zorder=2, markeredgecolor='darkred', 
                markeredgewidth=1.5)
    
    # Formatting
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('2D Spline Visualization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig, ax



def plot_xy_trajectory(xs, ys, ax=None, label="Trajectory", linestyle='-'):
    """
    xs: 1D array of x positions
    ys: 1D array of y positions
    ax: optional matplotlib Axes to plot on
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(xs, ys, linestyle=linestyle, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    return ax

#def system function
def make_fun(v_fn, omega_fn):
    def fun(t, y):
        x, y_pos, theta = y
        v = v_fn(t, y)
        omega = omega_fn(t, y)
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ])
    return fun
# define profiles (can depend on t and/or y)
# v_fn     = lambda t, y: 1.0 + 0.01*t
# omega_fn = lambda t, y: 0.1*t

# fun = make_fun(v_fn, omega_fn)

def solver(xs, ys, thetas, v, w, dt): #3 states, 2 control inputs, time step
    x_new = v*np.cos(thetas[-1])*dt + xs[-1]
    y_new = v*np.sin(thetas[-1])*dt + ys[-1]
    theta_new = w*dt + thetas[-1]
    xs.append(x_new)
    ys.append(y_new)
    thetas.append(theta_new)
    return xs, ys, thetas

t_0 = 0
t_f = 30
dt = 0.01
t = 0
xs, ys, thetas = [0],[0],[1.5]
v_fn     = lambda t: 1.0# + 0.01*t
omega_fn = lambda t: -1*np.sin(t)+0.01*t

while t<30:
    v = v_fn(t)
    w = omega_fn(t)
    solver(xs, ys, thetas, v, w, dt)
    t += dt


# t_eval = np.arange(t_0, t_f + dt, dt)
# init_state = np.array([0, 0, 0])

# plot_xy_trajectory(xs, ys)


# Example usage: for making splines
# Hard-coded points (no randomness)
points = [(0, 0), (2, 4), (8, 7), (10, 10)]
x_max = 10.0
y_max = 10.0

spline, x_pts, y_pts = generate_spline(points, spline_type="cubic")

# Create spline plot (returns fig, ax)
fig, ax = visualize_spline_2d(spline, x_pts, y_pts, x_max, y_max)

# Plot trajectory on the SAME axes
plot_xy_trajectory(xs, ys, ax=ax, label="ODE Trajectory", linestyle="--")

ax.legend()
plt.show()
