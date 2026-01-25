from tkinter import Y
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, UnivariateSpline


def print_results(results):
    xs, ys, thetas, x_dots, y_dots, theta_dots = results

    for i, (x, y, theta, xd, yd, thetad) in enumerate(
        zip(xs, ys, thetas, x_dots, y_dots, theta_dots)
    ):
        print(
            f"{i:4d}: "
            f"x = {x:.3f}, y = {y:.3f}, theta = {theta:.3f}, "
            f"x_dot = {xd:.3f}, y_dot = {yd:.3f}, theta_dot = {thetad:.3f}"
        )

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
    t_of_s = interp1d(arc_lengths, t_fine, kind='linear', 
                      bounds_error=False, fill_value=(0, 1))
    
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


def visualize_spline_2d(spline, x_points, y_points, x_max, y_max, 
                         n_eval_points=200, figsize=(10, 6), show_points=True):
    """
    Visualize a 2D spline curve with matplotlib.
    
    Parameters:
    -----------
    spline : scipy interpolation object or tuple (spline_x, spline_y)
        The spline function(s) to visualize. If tuple, assumes parametric spline.
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
    # Check if spline is parametric (tuple) or regular
    if isinstance(spline, tuple):
        # Parametric spline: evaluate at parameter values from 0 to 1
        spline_x, spline_y = spline
        t_eval = np.linspace(0, 1, n_eval_points)
        x_eval = spline_x(t_eval)
        y_eval = spline_y(t_eval)
    else:
        # Regular spline: evaluate at x values
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
    xs: 1D array or list of x positions (handles nested lists)
    ys: 1D array or list of y positions (handles nested lists)
    ax: optional matplotlib Axes to plot on
    """
    def flatten_to_array(data):
        """Convert nested lists/arrays to flat 1D array."""
        result = []
        for item in data:
            # Handle nested lists/arrays
            if isinstance(item, (list, tuple, np.ndarray)):
                # If it's a sequence, take first element or flatten
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
    
    xs = flatten_to_array(xs)
    ys = flatten_to_array(ys)

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


def plot_xy_trajectory_with_time(xs, ys, times=None, dt=None, ax=None, label="Trajectory", 
                                  colormap='viridis', linewidth=2, alpha=0.8):
    """
    Plot trajectory with time visualization using color mapping.
    
    Parameters:
    -----------
    xs : array-like
        1D array or list of x positions (handles nested lists)
    ys : array-like
        1D array or list of y positions (handles nested lists)
    times : array-like, optional
        Time values for each point. If None, will generate from dt or index.
    dt : float, optional
        Time step. Used to generate times if times is None.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    label : str
        Label for the trajectory
    colormap : str
        Matplotlib colormap name (default: 'viridis')
    linewidth : float
        Line width for the trajectory
    alpha : float
        Transparency of the line
    
    Returns:
    --------
    fig : matplotlib figure
        The figure object (only if ax is None)
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
    
    xs = flatten_to_array(xs)
    ys = flatten_to_array(ys)
    
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")
    
    n_points = len(xs)
    
    # Generate time array if not provided
    if times is None:
        if dt is not None:
            times = np.arange(0, n_points * dt, dt)[:n_points]
        else:
            times = np.arange(n_points)  # Use index as time
    
    times = np.asarray(times, dtype=float)
    if len(times) != n_points:
        # Truncate or pad times to match
        if len(times) > n_points:
            times = times[:n_points]
        else:
            times = np.pad(times, (0, n_points - len(times)), mode='edge')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=times.min(), vmax=times.max())
    
    # Plot trajectory with color mapping
    # Use LineCollection for smooth color transitions
    from matplotlib.collections import LineCollection
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(times)
    line = ax.add_collection(lc)
    
    # Add colorbar
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Time', rotation=270, labelpad=15)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{label} (colored by time)")
    
    return fig, ax


def plot_both_trajectories_with_time(ideal_xs, ideal_ys, ideal_times, 
                                     ode_xs, ode_ys, ode_times,
                                     ax=None, figsize=(12, 10),
                                     ideal_colormap='viridis', ode_colormap='plasma',
                                     ideal_label='Ideal Trajectory', ode_label='ODE Trajectory'):
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
        Colormap for ODE trajectory (default: 'plasma')
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
    lc_ideal = LineCollection(segments_ideal, cmap=cmap_ideal, norm=norm_combined, 
                              linewidth=2.5, alpha=0.7, label=ideal_label)
    lc_ideal.set_array(ideal_times)
    ax.add_collection(lc_ideal)
    
    # Plot ODE trajectory with time coloring
    points_ode = np.array([ode_xs, ode_ys]).T.reshape(-1, 1, 2)
    segments_ode = np.concatenate([points_ode[:-1], points_ode[1:]], axis=1)
    cmap_ode = plt.get_cmap(ode_colormap)
    lc_ode = LineCollection(segments_ode, cmap=cmap_ode, norm=norm_combined, 
                            linewidth=2, alpha=0.8, linestyle='--', label=ode_label)
    lc_ode.set_array(ode_times)
    ax.add_collection(lc_ode)
    
    # Add colorbar (using one of the collections for the scale)
    cbar = plt.colorbar(lc_ideal, ax=ax)
    cbar.set_label('Time', rotation=270, labelpad=15)
    
    # Format axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Ideal vs ODE Trajectory (colored by time)")
    ax.legend()
    plt.tight_layout()
    
    return fig, ax

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
        # Initialize x_dots and y_dots as lists (not numpy arrays) so we can append later
        self.x_dots = [self.v_dots[0] * np.cos(self.thetas[0])]
        self.y_dots = [self.v_dots[0] * np.sin(self.thetas[0])]


        #gives second position values for second derivative solver by using velocity
        self.xs.insert(0, (self.xs[0]-self.v_dots[0]*np.cos(self.thetas[-1])*self.dt))
        self.ys.insert(0, (self.ys[0]-self.v_dots[0]*np.sin(self.thetas[-1])*self.dt))
        self.thetas.insert(0, (self.thetas[0]-self.theta_dots[0]*self.dt))

        # self.x_dots.insert(0, 0)
        # self.y_dots.insert(0, 0)
        # self.theta_dots.insert(0, 0)

        print(self.xs, self.ys, self.thetas)



    def solve(self, f, f_theta, torque): #advances state by time value, returns states
        #position solves

         #force is a vector. change how you can apply force in different directions
        self.xs.append((f/self.m * np.cos(self.thetas[-1]) * self.dt*2) + 2*self.xs[-1]-self.xs[-2] )
        self.ys.append((f/self.m * np.sin(self.thetas[-1]) * self.dt*2) + 2*self.ys[-1]-self.ys[-2] )
        self.thetas.append((2 / self.m / self.r**2 * torque * self.dt**2 ) + 2*self.thetas[-1] - self.thetas[-2])

        # #vel solves
        # self.x_dots.append(f/self.m * np.cos(self.thetas[-1])*self.dt + self.x_dots[-1])
        # self.y_dots.append(f/self.m * np.sin(self.thetas[-1])*self.dt + self.y_dots[-1])
        # self.theta_dots.append(2 / self.m / self.r**2* torque *self.dt + self.theta_dots[-1])

        # print(len(self.xs), len(self.ys), len(self.x_dots), len(self.theta_dots))

        return self.xs, self.ys, self.thetas#, self.x_dots, self.y_dots, self.theta_dots

class controller:
    def __init__(self,traj, kd_lin= 0, kp_lin = 0, kd_ang= 0, kp_ang = 0, dt = 0.01):
        if traj is None:  
            traj = [[0.0, 0.0]]   
        self.traj = traj
        self.dt = dt
        self.kd_lin = kd_lin
        self.kp_lin = kp_lin
        self.kd_ang = kd_ang
        self.kp_ang = kp_ang

        
    def PD_control(self, targ, xs, ys, thetas): #Targ = [x, y, vel] state = [x, y, theta]       , x_dot, y_dot, theta_dot]
        # Position errors
        dx = targ[0]-xs[-1]
        dy = targ[1]-ys[-1]
        dist_error = np.sqrt(dx**2 + dy**2)

        v_x = (xs[-1]-xs[-2])/self.dt
        v_y = (ys[-1]-ys[-2])/self.dt
        theta_dot = (thetas[-1]-thetas[-2])/self.dt
        v_cur = np.sqrt(v_x**2+v_y**2)
        f = self.kp_lin * dist_error + self.kd_lin * (targ[2] - v_cur)
        torque =self.kp_ang * (np.arctan2(dy,dx)-thetas[-1]) - self.kd_ang*theta_dot
        return f, torque

'''MAINC ODE BEGINS HERE'''        
#initialize starting variables
t_0, t_f, t, dt = 0, 10, 0, 0.01
m, r = 1, 0.1
xs, ys, thetas = [], [], []
control_points = [(0, 0), (2, 4), (8, 7), (10, 10)]
x_max, y_max = 20.0, 20.0 #graph x and y axis max
v_traj = 1
#placeholder force and torque functions
f_fn     = lambda t: -1.0*np.sin(2*t)
alpha_fn = lambda t: -1*np.sin(t)+0.01*t 

f, torque = 0, 0
#initial state
state0 = [0, 0, 1.7, 0.5, 0]  #should give in xytheta, vel, ang_vel 
(spline_x, spline_y), traj = generate_spline(control_points, dt=dt, spline_type="cubic", v = v_traj)

solver = solver(m, r, dt, state0)
controller  = controller(traj, kd_lin= 0.0, kp_lin = 0.1, kd_ang= 0.0, kp_ang = 0.1, dt = dt )



#solver loop - track time for visualization
times = []
while t<t_f:
    # f = f_fn(t)
    # torque = alpha_fn(t)
    xs,  ys,  thetas = solver.solve(f,0, torque)    #force, force_angle, torque
    f, torque = controller.PD_control(traj, xs, ys, thetas)
    times.append(t)
    # xs,  ys,  thetas,  x_dots,  y_dots,  theta_dots
    t += dt 


#debugging
# print_results(results)


# Create spline plot (returns fig, ax)
pts = np.array(control_points)
fig, ax = visualize_spline_2d((spline_x, spline_y), pts[:, 0], pts[:, 1], x_max, y_max)

# Plot trajectory on the SAME axes (regular plot)
plot_xy_trajectory(xs, ys, ax=ax, label="ODE Trajectory", linestyle="--")

ax.legend()
plt.show()

# Create a combined plot with time visualization for both trajectories
# Extract ideal trajectory (spline) points
traj_xs = traj[:, 0]
traj_ys = traj[:, 1]
traj_times = np.arange(0, len(traj_xs) * dt, dt)[:len(traj_xs)]

# Extract ODE trajectory points and times
times_array = np.array(times, dtype=float)

# Plot both trajectories with time visualization
fig2, ax2 = plot_both_trajectories_with_time(
    ideal_xs=traj_xs,
    ideal_ys=traj_ys,
    ideal_times=traj_times,
    ode_xs=xs,
    ode_ys=ys,
    ode_times=times_array
)
plt.show()
