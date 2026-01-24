from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, UnivariateSpline



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
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
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
class solver:
    def __init__(self, m = 1, r=0.01, dt = 0.01, state0 = None):  #change so you can pass in init pos and init vel
        #state0: [x_0, y_0, theta_0, vel_0, theta_dot_0]

        if state0 is None:  
            state0 = [0.0, 0.0, 0.0, 0.0, 0.0]        
        
        self.m = m
        self.r = r
        self.dt = dt
        self.xs, self.ys, self.thetas = [[v] for v in state0[:3]]
        self.v_dots, self.theta_dots = [[v] for v in state0[3:]]

        #gives second position values for second derivative solver by using velocity
        self.xs.insert(0, (self.xs[0]-self.v_dots[0]*np.cos(self.thetas[-1])*self.dt))
        self.ys.insert(0, (self.ys[0]-self.v_dots[0]*np.sin(self.thetas[-1])*self.dt))
        self.thetas.insert(0, (self.thetas[0]-self.theta_dots[0]*self.dt))

        print(self.xs, self.ys, self.thetas)



    def solve(self, f, f_theta, torque): #advances state by time value, returns states
        #position solves

         #force is a vector. change how you can apply force in different directions
        self.xs.append((f/self.m * np.cos(self.thetas[-1]) * self.dt*2) + 2*self.xs[-1]-self.xs[-2] )
        self.ys.append((f/self.m * np.sin(self.thetas[-1]) * self.dt*2) + 2*self.ys[-1]-self.ys[-2] )
        self.thetas.append((2 / self.m / self.r**2 * torque * self.dt**2 ) + 2*self.thetas[-1] - self.thetas[-2])

        # #vel solves
        # self.x_dots.append(f/self.m * np.cos(self.thetas[-1])*self.dt + self.x_dots[-1])
        # self.x_dots.append(f/self.m * np.sin(self.thetas[-1])*self.dt + self.y_dots[-1])
        # self.theta_dots.append(1/2*self.m*self.r**2* alpha *self.dt + self.theta_dots[-1])

        return self.xs, self.ys, self.thetas#, self.x_dots, self.y_dots, self.theta_dots

t_0 = 0
t_f = 30
t = 0
m = 1
r = 0.01
dt = 0.01
xs = []
ys = []
thetas = []

#placeholder force and torque functions
f_fn     = lambda t: -1.0*np.sin(2*t)
alpha_fn = lambda t: -1*np.sin(t)+0.01*t 

#initial state
state0 = [0, 0, 0.7, 2, 0]  #should give in xytheta, vel, ang_vel 

solver = solver(m, r, dt, state0)

while t<10:
    f = f_fn(t)
    torque = alpha_fn(t)
    results = solver.solve(f,0, torque)    #force, force_angle, torque
    # results = solver.solve(0.01, 0, 0)   #force, force_angle, torque

    xs = results[0]
    ys = results[1]
    thetas = results[2]
    t += dt

for i, (x, y, theta) in enumerate(zip(xs, ys, thetas)):
    print(f"{i:4d}: x = {x:.3f}, y = {y:.3f}, theta = {theta:.3f}")

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
