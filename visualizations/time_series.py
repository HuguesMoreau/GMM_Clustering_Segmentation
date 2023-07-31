import itertools
import warnings
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import softmax, logsumexp

from utils.data_generation import generate_data


#%%
def generate_colors(n_colors:int, method:str="random"):
    """
    Parameters
    ----------
    n_colors: int
    method: str, one of:
        "random" (default) uniform values in [0,1]
        "basic colors": Red, Green, Blue, Cyan, Magenta, etc.
        "spherical": generate points on a unit sphere

    Returns
    -------
    colors: np array of shape (n_colors, 3) which elements are in [0,1]
    """

    if method == "basic colors":
        assert (n_colors < 9), f"Only 8 basic colors are available, asked for {n_colors}"
        colors = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.9, 0.9, 0.0],
                           [0.9, 0.0, 0.9],
                           [0.0, 0.9, 0.9],
                           [0.0, 0.0, 0.0],
                           [0.5, 0.5, 0.5] ])
        return colors[:n_colors,:]
    elif method == "random":
        return np.random.uniform(0,1,(n_colors,3))
    elif method == "spherical":
        if n_colors <= 12:
            phi = (1 + np.sqrt(5))/2
            isocaedron =  np.array([[ 1.0,  phi,  0.0],
                                    [ 1.0, -phi,  0.0],
                                    [-1.0,  phi,  0.0],
                                    [-1.0, -phi,  0.0],
                                    [ 0.0,  1.0,  phi],
                                    [ 0.0,  1.0, -phi],
                                    [ 0.0, -1.0,  phi],
                                    [ 0.0, -1.0, -phi],
                                    [ phi,  0.0,  1.0],
                                    [-phi,  0.0,  1.0],
                                    [ phi,  0.0, -1.0],
                                    [-phi,  0.0, -1.0],
                                    ])
            isocardron_in_unit_interval = 0.5*(isocaedron + np.ones((1,3)) )
            return isocardron_in_unit_interval[:n_colors,:]
        else:
            colors_zero_centered = np.random.uniform(-1,1,(n_colors,3))
            colors_zero_sphere = colors_zero_centered / np.linalg.norm(colors_zero_centered, axis=1, keepdims=True)
            return 0.5*(colors_zero_sphere + np.ones((1,3)) )
    else:
        raise ValueError(f"Unknown method of choosing colors ('{method}'). Should be one of 'basic colors', 'random', 'spherical'")




if __name__ =="__main__":
    for n_colors in range(1,8+1):
        for method in ['basic colors', 'random', 'spherical']:
            generate_colors(n_colors, method)



#%%
def plot_time_series_one_segment(x, r, u, v, plot_priors=False, plot_responsibilities=False,
                                 color_per_segment=None, origin=None, *args, **kwargs):
    """
    Parameters
    ----------
    x: np array with shape (n_days)
        ONE univariate time series from the data
    r: np array with shape (n_days, n_segments)
    u: np array with shape (n_segments,)
    v: np array with shape (n_segments,)
    plot_responsibilities: bool, optional
    plot_priors: bool, optional
    color_per_segment : np array with shape(n_segments, 3), optional
        If None (default), generate random colors
    origin: datetime.datetime or None
        Serves to set the legend on the x axis
    args, kwargs: arguments for the pls.plot() method

    Returns
    -------
    None.

    """


    n_days, n_segments = r.shape
    t = np.linspace(0, 1, n_days)

    t_axis = t if (origin == None) else np.array([origin + datetime.timedelta(days=i) for i in range(n_days)])
    n_plots = 1 + plot_responsibilities + plot_priors
    i_plot = 1

    if color_per_segment is None: color_per_segment = generate_colors(n_segments)

    if plot_priors:
        plt.subplot(n_plots, 1, i_plot)
        plt.ylabel(r'$\kappa_t^{k, s}$')
        p = softmax(t[:,np.newaxis] * u[np.newaxis,:] + v[np.newaxis,:], axis=1)  # shape: (n_days, n_segments)
        for s in range(n_segments):
            plt.plot(t_axis, p[:,s], c=color_per_segment[s,:])
        plt.grid(True)
        i_plot += 1
        plt.xticks([], [])


    if plot_responsibilities:
        plt.subplot(n_plots, 1, i_plot)
        plt.ylabel('responsibilities $r_{i,t}^{k, s}$')
        for s in range(n_segments):
            plt.scatter(t_axis, r[:,s], c=[color_per_segment[s,:]])
        plt.grid(True)
        i_plot += 1
        plt.xticks([], [])



    plt.subplot(n_plots, 1, i_plot)
    r_no_nan = r.copy()
    r_no_nan[np.isnan(r_no_nan)] = 0.
    for i_day in range(n_days-1):
        this_r = (r_no_nan[i_day,:] + r_no_nan[i_day+1,:])/2  # shape: n_segments
        this_color = this_r @ color_per_segment
        this_color = np.clip(this_color, 0., 1.) # rounding errors
        plt.plot([t_axis[i_day], t_axis[i_day+1]], [x[i_day], x[i_day+1]], c=this_color*1.0,
                 *args, **kwargs)
    plt.grid(True)


    return



if __name__ == "__main__":



    n_individuals, n_days = 1, 1000

    t = np.linspace(0,1, n_days)
    n_segments = 4
    ind_variables = np.zeros((n_individuals,1,0))
    temp_variables = np.sin(2*np.pi*5*t).reshape(1,-1,1)
    mixed_variables = np.zeros((n_individuals,n_days,0))
    variables =(ind_variables, temp_variables, mixed_variables)

    pi = np.ones(1)

    min_slope = 8.8 / 0.1
    u = np.linspace(-n_segments/2, n_segments/2, n_segments) * min_slope
    delta_u = u[1] - u[0]
    borders = [0.25, 0.5, 0.62]
    v = np.zeros(n_segments)
    for s in range(n_segments-1): v[s+1] = v[s] - borders[s] *(u[s+1]-u[s])
    # v -= np.mean(v)
    u, v = u[np.newaxis,:], v[np.newaxis,:]

    print(np.diff(v/u))

    m = np.array([0, -3, 0, 1]).reshape(1,1,4)
    alpha = np.zeros((0,1,1,4))
    beta  = np.array([0, 0, 0, -3]).reshape(1,1,1,4)
    gamma = np.zeros((0,1,1,4))
    contributions = (alpha, beta,gamma)
    sigma = np.array([0.5, 0.5, 5, 0.5]).reshape(1,1,4)


    Y, r = generate_data(pi, u, v, m, contributions, sigma, variables, covariance_type="diag")


    fig, axs = plt.subplots(3,1, figsize=(6,8))
    plot_time_series_one_segment(Y[0,:,0], r[0,:,0,:], u[0,:], v[0,:], plot_priors=True, plot_responsibilities=True,
                                     color_per_segment=None)
    axs[1].remove()
    plt.subplot(3,1,2)
    plt.plot(t, temp_variables.reshape(-1))
    plt.xticks([],[])
    plt.grid(True)
    plt.ylabel("time covariate")

    plt.savefig("results/figures/data_example.pdf")



    # params = [pi, u, v, m, alpha, beta, gamma, sigma]

    # we do not import at the beginning of the file to avoid dependency issues












