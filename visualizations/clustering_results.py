import itertools
import warnings
import numpy as np
from sklearn.manifold import MDS, TSNE  # needed for reordering the individuals
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from visualizations.time_series import generate_colors


all_markers = {k:v for (k,v) in Line2D.markers.items() if v != "nothing"}
del all_markers[","] # a pixel is too small for our taste
all_markers_list = list(all_markers.keys())
np.random.shuffle(all_markers_list)

#%%





def plot_mean_covar(mu, cov, c=[0,0,0], mean_label=None, sigma_label=None):
    """
    here, the dimension is assumed to be 2
    """
    # plotting the mean is easy
    plt.scatter(mu[0], mu[1], marker='+', s=50, c=[c], label=mean_label)
       # the [c] is here because matplotlib raises a warniong id c.shape = (3,)

    # plotting the ellipse covariance is harder
    eigenvariances, eigenvectors = np.linalg.eig(cov)

    i_max = np.argmax(eigenvariances) # index of the highest eigenvalue
    i_min = 1-i_max                   # index of the lowest eigenvalue

    # angle of the highest variance component
    dot_product = np.dot(eigenvectors[:,i_max], np.array([1,0])) # both eigenvectors are unitary, no need to normalize
    angle_radians = np.arccos(np.clip(dot_product, -1,1))   # between 0 and pi
    if np.dot(eigenvectors[:,i_max], np.array([0,1])) <0: angle_radians *= -1  # we want the angle to be in [-pi, pi]

    angle_degrees = angle_radians * 180/np.pi
    ellipse = matplotlib.patches.Ellipse((mu[0], mu[1]),
                 height=4*np.sqrt(eigenvariances[i_min]), width=4*np.sqrt(eigenvariances[i_max]),  # both eigenvariances are positive
                 angle = angle_degrees, edgecolor=c, facecolor=[0,0,0,0],     # adding alpha to the color
                 label=sigma_label, zorder=100)   #  the ellipsis is in front of anything else

    ax = plt.gca()
    ax.add_patch(ellipse)



if __name__ == '__main__':
    plt.figure(figsize=(8,8))  # we want a square
    Y = np.random.randn(1000, 2) @  np.random.randn(2, 2)
    plt.scatter(Y[:,0], Y[:,1], s=4)
    plot_mean_covar(mu=np.mean(Y, axis=0), cov=np.cov(Y, rowvar=False), c=[0.5, 0.5, 0.5])








#%%


def dummy_rectangle(**kwargs):
    """ Used for matplotlib legends """
    return Rectangle((0,0), 1,1, **kwargs)


def plot_2D_clustering(Y, r, ind_variables, temp_variables, colors_per_cluster=None, title="", new_figure=True):
    """
    Parameters
    ----------
    Y: np array with shape (n_individuals, n_days, dim)
        If dim is higher than 2, we plot the first two dimensions and merely raise a warning
    r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
    ind_variables:  np array with shape (n_individuals, n_ind_variables)
    temp_variables: np array with shape (n_days,        n_temp_variables)
    colors_per_cluster: np array with shape (n_clusters, 3), optional
        If omitted, random colors are generated
    title: str, optional
    new_figure: bool, defaults to True

    returns None
    """
    n_ind, n_days, n_clusters, n_segments = r.shape
    dim = Y.shape[2]  # dim can be higher than 2, in which case we plot only the first two dims
    marker_cycle = itertools.cycle(all_markers_list)

    if colors_per_cluster is None:
        colors_per_cluster = generate_colors(n_clusters*n_segments, method="random")

    if new_figure: plt.figure()
    plt.title(title)
    if ind_variables.dtype in ['int64', 'int32']:  # discrete variables
        if ind_variables.shape[1] + temp_variables.shape[1] == 0: return

        unique_ind_var  = np.unique(ind_variables,  axis=0)  # shape: (n_unique_ivar, n_ind_variables)
        unique_temp_var = np.unique(temp_variables, axis=0)  # shape: (n_unique_tvar, n_temp_variables)
        n_unique_ivar, n_unique_tvar = unique_ind_var.shape[0], unique_temp_var.shape[0]
        unique_ind_var  =  unique_ind_var[np.lexsort( unique_ind_var.T),:]  # we sort so that the marker order stays the same across runs
        unique_temp_var = unique_temp_var[np.lexsort(unique_temp_var.T),:]
        # n_var_combinations = n_unique_ivar * n_unique_tvar
        for i_ind_comb in range(n_unique_ivar):
            this_ind_vars = unique_ind_var[i_ind_comb,:]
            kept_ind = (ind_variables == this_ind_vars[np.newaxis,:]).all(axis=1)  # shape: (n_individuals,)

            for i_temp_comb in range(n_unique_tvar):
                this_temp_vars = unique_temp_var[i_temp_comb,:]
                kept_days = (temp_variables == this_temp_vars[np.newaxis,:]).all(axis=1)  # shape: (n_days,)

                kept_samples = (Y[kept_ind,:,:][:,kept_days,:]).reshape(-1, dim)
                kept_r = (r[kept_ind,:,:,:][:,kept_days,:,:]).reshape(-1, n_clusters*n_segments)
                colors_per_sample = np.clip(kept_r @ colors_per_cluster, 0,1)
                plt.scatter(kept_samples[:,0], kept_samples[:,1], c=colors_per_sample,
                            marker=next(marker_cycle),
                            label=f"samples (ind_var = {this_ind_vars}, temp_var = {this_temp_vars})")

    else: # continuous variables
        colors_per_sample = np.clip(r.reshape(n_ind*n_days, n_clusters*n_segments) @ colors_per_cluster, 0,1)
        plt.scatter((Y.reshape(-1, dim))[:,0], (Y.reshape(-1, dim))[:,1], c=colors_per_sample,
                    label="samples")

    plt.legend()











#%%


def plot_cluster_ind_day(r, colors_per_cluster=None, title=None, ind_ordering="TSNE", legend=False,
                         days_legend={}, displayed_periods={}, new_figure=True, ylabel=None):
    """
    /!\ in this function, we give the name "clusters" to both individual clusters and
    segments (day clusters) because the distinction does not matter.

    Parameters
    ----------
    r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
        such that r.sum(axis=(2,3)) == 1
        the soft assignement
    colors_per_cluster: np array with shape (n_clusters, 3), optional
        if not provided, a new set of colors is generated at random
    title: str, optional
    ind_ordering: str, optional
        One of "TSNE", "MDS", "majority_cluster", or "original"
        How to reorder the individuals so that the clusters look consistent
        Defaults to "TSNE" (the method that looks best on toy examples)
        To leave the order of the individuals identical to what it is in r, use "original"
    legend: bool, optional
        Whether matplotlib
        matplotlib displays the legend on top of the graph, thereby hiding a fraction of
        the figure. This is why it defaults to False
    days_legend: (optional) dict with
        keys: index, int within [0, n_days[
        value: datetime.datetime object
        Used to know which columns correspont to what date. Can be omitted to display nothing
    displayed_periods: dict, with:
        keys: couple of ints (number of days)
        values: couple of:
            color as accepted by matplotlib (string or triple)
            ticklabel: string
    new_figure: bool, defaults to True
        if False, plots on plt.gca()
    ylabel (string): defaults to None

    Returns
    -------
    None.

    (This function plots the graph on the current figure or subplot)


    """
    n_individuals, n_days, n_clusters, n_segments = r.shape

    missing_ind_days = np.any(np.isnan(r), axis=(2,3))  # shape: (n_indibiduals, n_days)
    if missing_ind_days.any():
        r = r.copy()  # do not modify the original r :)
        r[np.isnan(r)] = 1/(n_clusters*n_segments)  # replace NaNs with the mean
        # we will recolor these with grey


    # Individual reordezring: reorder by cluster first, segment second.
    # To do so, we create columns for the cluster that inctrase the difference between clusters;
    # and leave the difference between segments unchanged
    per_cluster_responsibility = np.sum(r, axis=3, keepdims=True) # shape: (n_individuals, n_days, n_clusters, 1)
    per_cluster_responsibility *= n_segments*10  # accentuate the differences between clusters
    sorting_values = np.concatenate([r, per_cluster_responsibility], axis=3)
    sorting_values = sorting_values.reshape(n_individuals, -1) # shape: (n_individuals, n_days*n_clusters*(n_segments+1))
    if sorting_values.shape[0] < sorting_values.shape[1]: # if more 'features' than samples
        sorting_values = np.concatenate([r, per_cluster_responsibility], axis=3) # shape: (n_individuals, n_days, n_clusters, n_segments)
        sorting_values = np.mean(sorting_values, axis=1)   # mean on days
        sorting_values = sorting_values.reshape(n_individuals, -1) # shape: (n_individuals, n_clusters*(n_segments+1))

    if ind_ordering == "TSNE":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # we do not care about default parameters changing
            tsne = TSNE(n_components=1,\
                        init='pca', learning_rate="auto")  # the arguments on the second line are only to shut some warnings up
            individual_embedding = tsne.fit_transform(sorting_values).reshape(-1)
        individual_order = np.argsort(individual_embedding)

    elif ind_ordering == "MDS":
        mds = MDS(n_components=1, metric=False, n_init=100) # we only care about the ordering of the individuals
        individual_embedding = mds.fit_transform(sorting_values).reshape(-1)
        individual_order = np.argsort(individual_embedding)

    elif ind_ordering == "greedy":
        individual_order = []
        current_ind = 0
        remaining_individuals = list(range(n_individuals))
        while len(remaining_individuals) > 0:
            dist_to_current_ind = lambda x: np.linalg.norm(sorting_values[current_ind,:]-sorting_values[x,:], axis=0)
            current_ind = min(remaining_individuals, key=dist_to_current_ind)
            individual_order.append(current_ind)
            remaining_individuals.remove(current_ind)
        individual_order = np.array(individual_order)

    elif ind_ordering == "majority_cluster":
        rho_ik = np.nanprod(np.nansum(r, axis=3), axis=1)  # shape: (n_individuals, n_clusters)
        closest_cluster = np.argmax(rho_ik, axis=1)        # shape: (n_individuals, n_clusters)
        individual_order = np.argsort(closest_cluster)

    elif ind_ordering == "original":
        individual_order = np.arange(n_individuals)

    if new_figure: plt.figure()

    # other esthetic arguments
    if colors_per_cluster is None:
        colors_per_cluster = np.random.uniform(0,1, (n_clusters*n_segments, 3))
    if title is not None:
        plt.title(title)

    color_image = r[individual_order,:,:,:].reshape(n_individuals, n_days, n_clusters*n_segments) @ colors_per_cluster
    color_image[missing_ind_days,:] = np.array([0.5, 0.5, 0.5])
    if not (ylabel is None):
        plt.yticks([], [])
        plt.ylabel(ylabel, fontsize=12)
    plt.imshow(np.clip(color_image, 0., 1.))



    if legend:
        legend_colors = []
        legend_names = []
        for k in range(n_clusters):
            for s in range(n_segments):
                i_color = k * n_segments + s
                rectangle = matplotlib.patches.Rectangle((0,0), 1, 1, facecolor=colors_per_cluster[i_color,:], linewidth=0)
                legend_colors.append(rectangle)
                legend_names.append(f"Cluster {k}, segment {s}")
        plt.legend(legend_colors, legend_names)

    # y ticks
    if ind_ordering == "majority_cluster":
        n_per_cluster, _ = np.histogram(closest_cluster, bins=np.arange(n_clusters+1)-0.5)
        n_per_cluster_0 = np.concatenate([np.array([0]), n_per_cluster], axis=0)
        cumsum_n_per_cluster = np.cumsum(n_per_cluster_0)

        ytick_locations = (cumsum_n_per_cluster[:-1]  + cumsum_n_per_cluster[1:])/2
        cluster_labels = [f"cluster {k+1} " for k in range(n_clusters)]
        ax = plt.gca()
        ax.set_yticks(ytick_locations, cluster_labels,
                      fontsize=8, va="center", ha="right", minor=True)
        ax.set_yticks(cumsum_n_per_cluster-5, minor=False)  # minus five is for visuals
        ax.tick_params(which='minor', length=0)


    old_x, old_y = plt.gca().get_xlim(), plt.gca().get_ylim()
    tick_positions, tick_labels = [], []
    for (day_start, day_end), (color, label) in displayed_periods.items():
        plt.plot([day_start, day_start], [0, n_individuals], c=color)
        plt.plot([day_end,   day_end],   [0, n_individuals], c=color)
        tick_labels.append(label)
        tick_positions.append((day_start+day_end)/2)
    if len(displayed_periods) > 0:
        # plt.tick_params(axis="x", which="minor", top=True, bottom=False)
        plt.gca().tick_params(axis='x',which='major',top=False,bottom=True, labeltop=False, labelbottom=True)
        plt.gca().tick_params(axis='x',which='minor',top=True,bottom=False, labeltop=True, labelbottom=False)
        plt.xticks(tick_positions, tick_labels, minor=True,
                   rotation=35, ha='left')

    # x ticks
    if len(days_legend) > 0:
        pos_ticks = list(days_legend.keys())
        date_ticks =  [date.strftime('%Y-%m-%d') for date in days_legend.values()]
        plt.xticks(pos_ticks, date_ticks, rotation=-20, ha='left')


    plt.gca().set_xlim(*old_x)
    plt.gca().set_ylim(*old_y)



if __name__ == "__main__":
    n_individuals = 100
    n_days = 100
    n_clusters = 4
    n_segments = 3
    days = np.arange(n_days)




    cluster_borders = np.random.choice(np.arange(n_individuals), n_clusters-1)
    cluster_borders = np.sort(cluster_borders)
    cluster_borders = np.concatenate([np.array([0]), cluster_borders, np.array([n_individuals])], axis=0)
    individual_shuffled = np.random.permutation(np.arange(n_individuals))

    r = np.zeros((n_individuals, n_days, n_clusters, n_segments))
    # hard assignation for the clusters, soft assignation for the segments
    for k in range(n_clusters):
        cluster_centers = np.random.choice(np.arange(n_days), n_segments).reshape(1,1,1,-1)
        days = days.reshape(1,-1,1,1)
        assignation = np.exp(-((days - cluster_centers+0.5)/(n_days**0.3))**2)   # shape: (1, n_days, 1, n_segments)
        assignation /= np.sum(assignation, axis=3, keepdims=True)                # shape: (1, n_days, 1, n_segments)
        this_cluster_individuals = individual_shuffled[cluster_borders[k]:cluster_borders[k+1]]
        r[this_cluster_individuals,:,k:k+1,:] = assignation
    assert np.allclose(r.sum(axis=(2,3)), 1)

    base_title = f'random clusters\n({n_clusters} clusters, {n_segments} segments)\n'
    colors_per_cluster = np.random.uniform(0,1, (n_clusters*n_segments, 3))
    plt.figure(figsize=(13,13))
    for i, ind_ordering in enumerate(["TSNE", "MDS", "greedy", "original"]):
        plt.subplot(2,2,i+1)
        plot_cluster_ind_day(r, colors_per_cluster, title=base_title + f"ind_ordering='{ind_ordering}'", ind_ordering=ind_ordering, new_figure=False)
    plt.tight_layout()






    n_clusters = 3

    r = np.zeros((n_individuals, n_days, n_clusters, 1))
    cluster = np.random.randint(0, n_clusters, size=(n_individuals, n_days))   # hard assignation for everyone
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r[i,j,cluster[i,j],0] = 1

    base_title = f'test with random clusters\n({n_clusters} clusters)\n'
    colors_per_cluster = np.random.uniform(0,1, (n_clusters, 3))
    plt.figure(figsize=(13,13))
    for i, ind_ordering in enumerate(["TSNE", "MDS", "greedy", "original"]):
        plt.subplot(2,2,i+1)
        plot_cluster_ind_day(r, colors_per_cluster, title=base_title + f"ind_ordering='{ind_ordering}'", ind_ordering=ind_ordering, new_figure=False)
    plt.tight_layout()






#%%


def is_dummy_variable(a):
    if np.isnan(a).all(): return True
    a_non_nan = a[~np.isnan(a)]
    h, _ = np.histogram(a_non_nan.reshape(-1), bins=100)
    n_nonzero_bins = np.sum( (h > 0).astype(int))
    # we accept three nonzero bins because the missing values are set to the mean,
    # which can reate a third value
    return n_nonzero_bins <= 3










def plot_model_mean(m, contributions, sigma, r, variables, variables_to_plot='all',
                    colors_per_cluster=None, variable_names=None, ylabel=None):
    """
    Parameters
    ----------
    m: np array with shape (dim, n_cluster, n_segments)
        the per-cluster mean
    contributions: triple of
        alpha: np array with shape (n_ind_variables,   dim, n_clusters, n_segments)
        beta:  np array with shape (n_temp_variables,  dim, n_clusters, n_segments)
        gamma: np array with shape (n_mixed_variables, dim, n_clusters, n_segments)
    sigma: np array which shape is either:
        (n_clusters, n_segments):           spherical variance
        (dim, n_clusters, n_segments):      diagonal covariance
        (dim, dim, n_clusters, n_segments): full covariance
        (dim, dim):                         shared covariance
        we assume you won't plot a model with 48 clusters and diagonal covariance :)
    r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
        the cluster soft-assignment
            variables: triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables)
        Exogenous variables are used to compute a **per-cluster** mean and standard deviation.
        The variables in each cluster might not keep the same distribution as the global distribution
        (especially of the variables and the cluster/segment assignment are not independant).
    variables_to_plot: list of strings, or "all" (default)
        We only show the effect of the variables which names are in this list, or alll of them if "all".
    colors_per_cluster: np array with shape (n_clusters*n_segments, 3), optional
        if not provided, a new set of colors is generated at random
    variable_names: dict, optional
        keys: "individual" and "temporal"
        values: list of strings
            the order of the strings matches the order of the columns in the variable matrices.
        If absent, the varoiable names are replaced with ind_variable_0, ind_variable_1, temp_variable_0, etc.
    ylabel: string, optional

    Returns
    -------
    None.

    (This function creates new figures)
    """

    alpha, beta, gamma = contributions
    ind_variables, temp_variables, mixed_variables = variables

    dim, n_clusters, n_segments = m.shape
    if n_segments == 1:  # swap cluster and segments, to plot everything on a single figure
        multiple_segments = False
        n_clusters, n_segments = n_segments, n_clusters
        m = m.transpose(0,2,1)
        r = r.transpose(0,1,3,2)
        alpha = alpha.transpose(0,1,3,2)
        beta  =  beta.transpose(0,1,3,2)
        gamma = gamma.transpose(0,1,3,2)
        if sigma.shape != (dim, dim):
            ndims = len(sigma.shape)
            sigma = np.transpose(sigma, list(range(ndims-2)) + [ndims-1, ndims-2])  # invert the last new dimentions
    else:
        multiple_segments = True

    if dim == 48: # CER Dataset (I know):
        dim_array = np.linspace(0,24,dim+1)[1:]   # 0.5, 1.0, 1.5, ..., 23.5, 24.0   all included
    else:
        dim_array = np.arange(1,dim+1)

    n_ind_variables   = alpha.shape[0]
    n_temp_variables  =  beta.shape[0]
    n_mixed_variables = gamma.shape[0]
    if variable_names is None:
        variable_names = {"individual":[ f"ind_variable_{i}"  for i in  range(n_ind_variables)],
                          "temporal"  :[f"temp_variable_{i}"  for i in range(n_temp_variables)],
                          "mixed"     :[f"mixed_variable_{i}" for i in range(n_mixed_variables)] }
    if variables_to_plot == "all":
        variables_to_plot = sum(variable_names.values(), [])
    else:
        assert (variable_names != None), "variable names must be provided to know which ones to plot"
        for v_name in variables_to_plot: assert v_name in sum(variable_names.values(), []), f"variable to plot '{v_name}' absent from variable names"
    n_columns = len(variables_to_plot) +1




    if colors_per_cluster is None:
        colors_per_cluster = np.random.uniform(0,1, (n_clusters*n_segments, 3))

    if sigma.shape == (dim, dim, n_clusters, n_segments):
        dims = np.arange(dim)
        diag_sigma = sigma[dims, dims, :, :]                            # shape: (dim, n_clusters, n_segments)
        n_columns += 1
    else:
        if sigma.shape == (n_clusters, n_segments):
            diag_sigma = sigma[np.newaxis,:,:] * np.ones((dim, 1))      # shape: (dim, n_clusters, n_segments)
        elif sigma.shape == (dim, n_clusters, n_segments):
            diag_sigma = sigma                                          # shape: (dim, n_clusters, n_segments)
        elif sigma.shape == (dim, dim):
            diag_sigma = np.diag(sigma)[:,np.newaxis,np.newaxis] * np.ones((1, n_clusters, n_segments))
                                                                        # shape: (dim, n_clusters, n_segments)



    ylim_min, ylim_max = np.inf, -np.inf
    subplot_list = []
    # Actual plot
    for k in range(n_clusters):
        plt.figure(figsize=(5*(1+n_columns), 5*n_segments))
        for s in range(n_segments):
            # =============================================================================
            # Computation of the mean : we need to estimate the mean contribution of the variables per cluster
            # =============================================================================
            epsilon = 1e-35
            this_cluster_weight = epsilon + r[:,:,k,s] / (np.nansum(r[:,:,k,s], axis=(0,1)) + epsilon)  # shape: (n_ind, n_days), weights sum to 1
            mu_per_sample = m[:,k,s][np.newaxis, np.newaxis,:]  +\
                (ind_variables   @ alpha[:,:,k,s]) +\
                (temp_variables  @  beta[:,:,k,s]) +\
                (mixed_variables @ gamma[:,:,k,s])  # shape: (n_individuals, n_days, dim)

            mu = np.nansum(this_cluster_weight[:,:,np.newaxis] * mu_per_sample, axis=(0,1)) # shape: (dim,)
            # the weights sum to one, so we compute a np.sum
            # if the weight summed to n_samples, we would have computed a np.mean instead of a np.sum

            this_subplot = plt.subplot(n_segments, n_columns, n_columns*s + 1)
            subplot_list.append(this_subplot)

            i_color = k * n_segments + s
            plt.plot(dim_array, mu,       c=colors_per_cluster[i_color,:], label=r"$\mu_{k,s}$")  # $E_{Y \in c}[\mathcal{N}]$
            plt.plot(dim_array, m[:,k,s], c=colors_per_cluster[i_color,:], label=r"$m_{k,s}$",  linestyle='dotted')

            plt.fill_between(dim_array, mu-np.sqrt(diag_sigma[:,k,s]), mu+np.sqrt(diag_sigma[:,k,s]),
                              facecolor=list(colors_per_cluster[i_color,:])+[0.2], label=r"$\mu_{k,s} \pm \sqrt{\Sigma_{k,s}}$")
            if s == n_segments - 1:
                plt.xlabel("data")
            else:
                plt.xticks([],[])
            if (ylabel is not None): plt.ylabel(ylabel)
            legend_title = f'Cluster {k}, segment {s}' if multiple_segments else f'Cluster {s}' # we swapped segments and clusters
            plt.legend(title=legend_title, fontsize=7)
            # ylims = plt.gca().get_ylim()
            # if ylims[0] < ylim_min: ylim_min = ylims[0]
            # if ylims[1] > ylim_max: ylim_max = ylims[1]


            plt.title(f'Cluster {k+1}, Segment {s+1}')




            # =============================================================================
            #   Adding the influence of variables
            # =============================================================================

            i_var_plot = 1  # zero is for the mean
            contributions_dict = {'individual':alpha, 'temporal':beta, 'mixed':gamma}
            for var_type, var_array in [('individual', ind_variables), ('temporal', temp_variables), ('mixed', mixed_variables)]:
                var_indices = [i for i in range(var_array.shape[2]) if variable_names[var_type][i] in variables_to_plot]
                for i_var in var_indices:
                    this_subplot = plt.subplot(n_segments, n_columns, n_columns*s + i_var_plot + 1)
                    subplot_list.append(this_subplot)


                    var_name = variable_names[var_type][i_var]
                    plt.title(var_name)
                    variable_per_sample = var_array[:,:,i_var] # shape: (n_individuals, 1), (1, n_days), or (n_individuals, n_days)

                    this_var_mean = np.nansum(this_cluster_weight * variable_per_sample, axis=(0,1))
                    variance_this_cluster = np.nansum(this_cluster_weight * (variable_per_sample - this_var_mean)**2 )  # empirical variance
                    std_this_cluster = np.sqrt(variance_this_cluster)
                    positive_var_contrib = contributions_dict[var_type][i_var,:,k,s] * std_this_cluster

                    if is_dummy_variable(variable_per_sample):
                        one_value, zero_value = np.nanmax(variable_per_sample), np.nanmin(variable_per_sample) # the variables might have been normalized
                        for (value, strvalue, marker) in [(one_value, "1", "+"), (zero_value, "0", "_")]:
                            is_value = (variable_per_sample == value) # shape: (n_individuals, 1)
                            weight_this_value = this_cluster_weight * is_value  # shape: (n_ind, n_days)
                            weight_this_value /= np.nansum(weight_this_value)
                            mean_this_value = np.nansum(weight_this_value[:,:,np.newaxis] * mu_per_sample, axis=(0,1))  # shape: (dim,)
                            plt.plot(dim_array, mean_this_value, marker+'-', c=colors_per_cluster[i_color,:],
                                      label=r"$y_{i," + str(i_var_plot+1) + r"}$ = "+strvalue, linewidth=0.5)

                    else:
                        label_contrib = r"3.std(y_{i," + str(i_var_plot+1) + r"}).\alpha_{" + str(i_var_plot+1) + ",k,s}$"
                        plt.plot(dim_array, mu,                                c=colors_per_cluster[i_color,:], label=r"$\mu_{k,s}$")
                        plt.plot(dim_array, mu + 3* positive_var_contrib, '+', c=colors_per_cluster[i_color,:], label=r"$\mu_{k,s} + "+label_contrib)
                        plt.plot(dim_array, mu - 3* positive_var_contrib, '_', c=colors_per_cluster[i_color,:], label=r"$\mu_{k,s} - "+label_contrib)





                    i_var_plot += 1
                    plt.legend()



            ylims = plt.gca().get_ylim()
            if ylims[0] < ylim_min: ylim_min = ylims[0]
            if ylims[1] > ylim_max: ylim_max = ylims[1]


    # now we have y_axis range for all segments of this cluster:
    for subplot in subplot_list:
        subplot.set_ylim(ylim_min, ylim_max)




if __name__ == "__main__":

    n_individuals = 1000
    n_days = 3*365
    dim = 48
    hours = np.linspace(0.5, 24., dim)
    hours_2 = np.stack([hours, hours], axis=-1)# shape: (dim=48, n_clusters=2)

    # We create two toy models

    # =============================================================================
    #  First model: all variables are i.i.d
    # =============================================================================
    # Clusters
    ind_grid, days_grid = np.meshgrid(np.linspace(-1,1,n_individuals), np.linspace(-1,1,n_days), indexing='ij')
    sigmoid = lambda x: 1/(1+np.exp(-x))
    r_cluster_1 = sigmoid(6*(ind_grid - 0.3*days_grid))                    # shape: (n_individuals, n_days)
    r = np.stack([r_cluster_1, 1.-r_cluster_1], axis=-1)[:,:,:,np.newaxis] # shape: (n_individuals, n_days, n_clusters=2, n_segments=1)

    # Effects on clusters on dimension
    m     = np.stack([np.cos(2*np.pi*hours/24),   np.sin(2*np.pi*hours/24)], axis=-1)[:,:,np.newaxis]  # shape (dim=48, n_clusters=2, n_segments=1)
        # the contribution of all variables is the same, only the variables' values will change
    alpha = np.array([np.sin(4*np.pi*hours_2/24)] * 3)[:,:,:,np.newaxis]  # shape: (n_ind_variables=3,  dim=48, n_clusters=2, n_segments=1)
    beta =  np.array([np.sin(4*np.pi*hours_2/24)] * 2)[:,:,:,np.newaxis]  # shape: (n_temp_variables=2, dim=48, n_clusters=2, n_segments=1)
    gamma = np.zeros((0, dim, 2,1))
    contributions = (alpha, beta, gamma)
    sigma = np.ones((2,1))

    ind_variables  = np.random.randn(n_individuals, 1, 3) * np.array([5, 0.2, 0]).reshape(1,1,3)
    ind_variables[:,0,2]  = np.random.randint(0, 2, size=(n_individuals,))
    temp_variables = np.random.randn(1, n_days, 2) + np.array([3, 0]).reshape(1,1,2)
    mixed_variables = np.zeros((n_individuals, n_days, 0))
    variables = (ind_variables, temp_variables, mixed_variables)
    variable_names = {"individual": ["high-variance variable",  "low-variance variable",  "dummy (one-hot) variable"],
                      "temporal":   ["variable with high mean", "variable with zero mean"],
                      "mixed":      []}

    plot_model_mean(m, contributions, sigma, r, variables,
                    colors_per_cluster=None, variable_names=variable_names)





    # =============================================================================
    #  Second model: variables distributions depend on cluster
    # =============================================================================
    m     = np.zeros((dim, 3,1))
    hours_3 = np.stack([hours, hours, hours], axis=-1)                 # shape: (dim=48, n_clusters=3)
    alpha = np.array([np.cos(2*np.pi*hours_3/24)])[:,:,:,np.newaxis]   # shape: (n_ind_variables =1, dim=48, n_clusters=3, n_segments=1)
    beta =  np.array([np.cos(4*np.pi*hours_3/24)])[:,:,:,np.newaxis]   # shape: (n_temp_variables=1, dim=48, n_clusters=3, n_segments=1)
    gamma = np.zeros((0, dim, 3,1))
    contributions = (alpha, beta, gamma)

    sigma = np.ones(3).reshape(-1,1)
    ind_variables  = 3*(np.linspace(0,1, n_individuals)[:,np.newaxis,np.newaxis]-0.25)   +\
                            0.2 * np.random.randn(n_individuals, 1,      1)
    temp_variables = 3*(np.linspace(0,1, n_days       )[np.newaxis,:,np.newaxis]-0.5)   +\
                            0.2 * np.random.randn(1,             n_days, 1)
    mixed_variables = np.zeros((n_individuals, n_days, 0))
    variables = (ind_variables, temp_variables, mixed_variables)

    variable_names = {"individual": ["'increasing' variable\n(higher in cluster 0)"],
                      "temporal":   [ "increasing variable \n(higher in cluster 1)"],
                      "mixed":      []}

    # Clusters
    ind_grid, days_grid = np.meshgrid(np.linspace(-1,1,n_individuals), np.linspace(-1,1,n_days), indexing='ij')
    sigmoid = lambda x: 1/(1+np.exp(-x))
    r_cluster_1 = sigmoid(6*ind_grid) # shape
    r_cluster_2_3 = 1-r_cluster_1
    r_cluster_2 = r_cluster_2_3 *   sigmoid(6*days_grid)     # shape: (n_individuals, n_days)
    r_cluster_3 = r_cluster_2_3 * (1 - sigmoid(6*days_grid)) # shape: (n_individuals, n_days)
    r = np.stack([r_cluster_1, r_cluster_2, r_cluster_3], axis=-1) # shape: (n_individuals, n_days, n_clusters=3)
    r = r[:,:,:,np.newaxis]                                  # shape: (n_individuals, n_days, n_clusters=3, n_segments=1)

    fig, ((ax_ind_var, ax_clusters), (ax_0, ax_temp_var)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 3], 'height_ratios':[3,1]},
                                                                         figsize=(12, 12))
    ax_clusters.imshow(r[:,:,:,0])
    ax_clusters.set_title("toy data: the colour represents the \nprobability of belonging to the each cluster ")

    ax_ind_var.set_ylabel("individuals")
    individuals = np.arange(n_individuals)
    ax_ind_var.plot(ind_variables[:,0,0], individuals, 'o', markersize=1, label=variable_names["individual"][0])
    ax_ind_var.invert_yaxis()
    ax_ind_var.legend()

    ax_temp_var.set_xlabel("days")
    days = np.arange(n_days)
    ax_temp_var.plot(days, temp_variables[0,:,0], label=variable_names["temporal"][0])
    ax_temp_var.legend()

    ax_0.set_xticks([],[])
    ax_0.set_yticks([],[])



    plot_model_mean(m, contributions, sigma, r, variables,
                    colors_per_cluster=None, variable_names=variable_names)




#%%


