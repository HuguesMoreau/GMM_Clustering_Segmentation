import itertools
import warnings
import numpy as np
from sklearn.manifold import MDS, TSNE  # needed for reordering the individuals
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from visualizations.clustering_results import is_dummy_variable

all_markers = {k:v for (k,v) in Line2D.markers.items() if v != "nothing"}
del all_markers[","] # a pixel is too small for our taste
all_markers_list = list(all_markers.keys())
np.random.shuffle(all_markers_list)




def plot_model_mean_unidimensional(m, contributions, sigma, r, variables, variables_to_plot="all",
                                   colors_per_cluster=None, variable_names=None, ylabel=None):
    """
    This function produces similar results to plot_model_mean, but is better adapted to
    single-variable data.

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
    assert dim == 1


    n_ind_variables   = alpha.shape[0]
    n_temp_variables  =  beta.shape[0]
    n_mixed_variables = gamma.shape[0]

    if variable_names is None:
        variable_names = {"individual":[ f"ind_variable_{i}" for i in   range(n_ind_variables)],
                          "temporal"  :[f"temp_variable_{i}"  for i in  range(n_temp_variables)],
                          "mixed"     :[f"mixed_variable_{i}" for i in range(n_mixed_variables)] }
    if variables_to_plot == "all":
        variables_to_plot = sum(variable_names.values(), [])
    else:
        assert (variable_names != None), "wariable names must be provided to know which ones to plot"
        for v_name in variables_to_plot: assert v_name in sum(variable_names.values(), []), f"variable to plot '{v_name}' absent from variable names"

    n_variables = len(variables_to_plot)

    if colors_per_cluster is None:
        colors_per_cluster = np.random.uniform(0,1, (n_clusters*n_segments, 3))

    # Actual plot
    plt.figure(figsize=(4*n_segments, 4*(n_clusters+1)))
    print("n_segments, n_clusters", n_segments, n_clusters)
    print("variables_to_plot,", variables_to_plot)

    # prepare to set the axes
    ylim_min, ylim_max = np.inf, -np.inf
    subplot_list = []
    for k in range(n_clusters):
        for s in range(n_segments):
            # =============================================================================
            # Computation of the mean : we need to estimate the mean contribution of the variables per cluster
            # =============================================================================
            epsilon = 1e-35

            this_cluster_weight = epsilon + r[:,:,k,s] / (np.nansum(r[:,:,k,s], axis=(0,1)) + epsilon)  # shape: (n_ind, n_days), weights sum to 1
            mu_per_sample = m[:,k,s][np.newaxis, np.newaxis,:]  +\
                    (ind_variables   @ alpha[:,:,k,s])  +\
                    (temp_variables  @  beta[:,:,k,s])  +\
                    (mixed_variables @ gamma[:,:,k,s])
            mu = np.nansum(this_cluster_weight[:,:,np.newaxis] * mu_per_sample, axis=(0,1)) # shape: (dim,)
            # the weights sum to one, so we compute a np.sum
            # if the weight summed to n_samples, we would have computed a np.mean instead of a np.sum

            subplot = plt.subplot(n_clusters+1, n_segments, k*n_segments + s + 1)
            subplot_list.append(subplot)
            plt.grid(True)
            i_color = k*n_segments + s
            plt.plot([-1.5, n_variables-0.5], [mu, mu], c='k')
            plt.errorbar(x=[-1], y=mu, yerr=np.sqrt(sigma[:,k,s]),
                             color=list(colors_per_cluster[i_color,:]))




            # =============================================================================
            #   Adding the influence of variables
            # =============================================================================
            var_names_ticklabels = ["Variance"]
            i_var_plot = 0

            contributions_dict = {'individual':alpha, 'temporal':beta, 'mixed':gamma}
            for var_type, var_array in [('individual', ind_variables), ('temporal', temp_variables), ('mixed', mixed_variables)]:
                var_indices = [i for i in range(var_array.shape[2]) if variable_names[var_type][i] in variables_to_plot]

                for i_var in var_indices:
                    variable_per_sample = var_array[:,:,i_var] # shape: (n_individuals, 1), (1, n_days), or (n_individuals, n_days)

                    this_var_mean = np.nansum(this_cluster_weight * variable_per_sample, axis=(0,1))
                    variance_this_cluster = np.nansum(this_cluster_weight * (variable_per_sample - this_var_mean)**2 )
                    std_this_cluster = np.sqrt(variance_this_cluster)
                    positive_var_contrib = contributions_dict[var_type][i_var,:,k,s] * std_this_cluster

                    var_names_ticklabels.append(variable_names[var_type][i_var])
                    if is_dummy_variable(variable_per_sample):
                        one_value, zero_value = np.nanmax(variable_per_sample), np.nanmin(variable_per_sample) # the variables might have been normalized

                        is_1 = (variable_per_sample == one_value) # shape: (n_individuals, 1)
                        weight_1 = this_cluster_weight * is_1  # shape: (n_ind, n_days)
                        if np.nansum(weight_1) < 1e-10:
                            mean_1 = mu
                        else:
                            weight_1 /= (np.nansum(weight_1) + 1e-10)
                            mean_1 = np.nansum(weight_1[:,:,np.newaxis] * mu_per_sample, axis=(0,1))  # shape: (dim,)

                        is_0 = (variable_per_sample == zero_value) # shape: (n_individuals, 1)
                        weight_0 = this_cluster_weight * is_0  # shape: (n_ind, n_days)
                        if np.nansum(weight_0) < 1e-10:
                            mean_0 = mu
                        else:
                            weight_0 /= (np.nansum(weight_0) + 1e-10)
                            mean_0 = np.nansum(weight_0[:,:,np.newaxis] * mu_per_sample, axis=(0,1))  # shape: (dim,)


                        # plt.bar(x=i_var, y=mu, height=mean_this_value-mu, facecolor=colors_per_cluster[i_color,:],
                        #          label=r"$a_{i," + str(i_ind_var+1) + r"}$ = "+strvalue, linewidth=0.5)
                        plt.plot([i_var_plot, i_var_plot], [mean_1, mean_0], c=colors_per_cluster[i_color,:])
                        plt.plot([i_var_plot], [mean_0], c=colors_per_cluster[i_color,:], marker='$-$')
                        plt.plot([i_var_plot], [mean_1], c=colors_per_cluster[i_color,:], marker='$+$')
                        plt.plot([i_var_plot], [mean_0], c='k', marker='$-$')
                        plt.plot([i_var_plot], [mean_1], c='k', marker='$+$')
                    else:
                        plt.plot([i_var_plot, i_var_plot], [mu-3*positive_var_contrib, mu+3*positive_var_contrib], c=colors_per_cluster[i_color,:])
                        plt.plot([i_var_plot], [mu-3*positive_var_contrib], c='k', marker='$-$')
                        plt.plot([i_var_plot], [mu+3*positive_var_contrib], c='k', marker='$+$')
                    i_var_plot += 1


                    plt.title(f'Cluster {k+1}, Segment {s+1}')
                    if k == n_clusters-1: # last row
                        plt.xticks(np.arange(-1, len(var_names_ticklabels)-1), var_names_ticklabels, ha='left', rotation=-45)
                    else:
                        plt.xticks(np.arange(-1, len(var_names_ticklabels)-1), len(var_names_ticklabels)*[' '], ha='left', rotation=-45)



            ylims = plt.gca().get_ylim()
            if ylims[0] < ylim_min: ylim_min = ylims[0]
            if ylims[1] > ylim_max: ylim_max = ylims[1]

        # now we have y_axis range for all segments of this cluster:
        for subplot in subplot_list:
            subplot.set_ylim(ylim_min, ylim_max)






if __name__ =="__main__":

    base_colors_per_cluster = np.array([[1.,  0.,  0.],
                                        [0.,  0.9, 0.],
                                        [0.,  0.,  1.],
                                        [0.9, 0.9, 0.],
                                        [1.,  0.7, 0.1],
                                        [0.9, 0.3, 0.9],
                                        [0.6, 0.,  0.6],
                                        [0.,  0.9, 0.9],
                                        [0.5, 0.5, 0.5],
                                        [0., 0., 0.],])

    base_colors_per_cluster = model.base_colors[::model.n_segments,:]
    if (model.n_segments > 1) and (X_train.shape[2] == 1) :


        variables_to_plot = ['school_holiday', 'national_holiday', "time"] + [vname for vname in  variable_names["temporal"] if "is_" in vname]

        contributions = (model.alpha, model.beta, model.gamma)
        plot_model_mean_unidimensional(model.m, contributions, model.sigma, np.exp(log_r_train), variables_train,
                                       variables_to_plot=variables_to_plot,
                                       # colors_per_cluster=base_colors_per_cluster[:model.n_clusters*model.n_segments,:],
                                       colors_per_cluster=model.base_colors[:model.n_clusters*model.n_segments,:],
                                       variable_names=variable_names, ylabel=None)


    elif (model.n_segments == 1):

        if X_train.shape[2] == 1:
            contributions = (model.alpha[:,:,np.newaxis,:,0], model.beta[:,:,np.newaxis,:,0], model.gamma[:,:,np.newaxis,:,0])
            plot_model_mean_unidimensional(model.m[:,np.newaxis,:,0], contributions, model.sigma[:,np.newaxis,:,0], np.exp(log_r_train)[:,:,np.newaxis,:,0], variables_train,
                                           variables_to_plot="all",
                                           colors_per_cluster=base_colors_per_cluster[:model.n_clusters,:], variable_names=variable_names, ylabel=None)




    if X_train.shape[2] > 1:
        plt.figure(figsize=(3*model.n_clusters, 3))

        ylim_min, ylim_max = np.inf, -np.inf
        subplot_list = []

        rotate = np.arange(7) #(np.arange(7)-1) % 7
        for k in range(model.n_clusters):
            for s in range(model.n_segments):
                subplot_list.append(plt.subplot(1, model.n_clusters, k+1))
            plt.plot(np.arange(7), model.m[rotate,k,:], "-+", c=base_colors_per_cluster[k,:])
            plt.fill_between(np.arange(7), model.m[rotate,k,:]-model.sigma[rotate,k,:], model.m[rotate,k,:]+model.sigma[rotate,k,:],
                             facecolor=base_colors_per_cluster[k,:], alpha=0.3)
            plt.grid(True)


            ylims = plt.gca().get_ylim()
            if ylims[0] < ylim_min: ylim_min = ylims[0]
            if ylims[1] > ylim_max: ylim_max = ylims[1]

        ylim_min = 0.3
        for subplot in subplot_list:
            subplot.set_ylim(ylim_min, ylim_max)


        #%%


    from visualizations.clustering_results import plot_cluster_ind_day, plot_model_mean, generate_colors
    plot_model_mean(model.m, model.alpha, model.beta, model.sigma, np.exp(log_r_train),
                    ind_variables_train, temp_variables_train,
                    colors_per_cluster=model.base_colors, variable_names=variable_names, ylabel='normalized log10(entries)')

#%%


from utils.nan_operators import  nan_logsumexp

def plot_proportion_of_each(log_r, ind_variables, variable_names, colors_per_cluster=None):
    """
    This function produces similar results to plot_model_mean, but is better adapted to
    single-variable data.

    Parameters
    ----------
    log_r: np array with shape (n_ind, n_days, n_clusters, n_segments)
    ind_variables: np array withb shape: (n_individuals, 1, n_ind_var)
    variable_names: list of strings or dictionnary with keys: "temporal", "individual", and "mixed"

    Returns
    -------
    None.

    (This function creates new figures)
    """

    if type(variable_names) != list:
        variable_names = variable_names["individual"]

    n_clusters = log_r.shape[2]
    proba_this_cluster_per_day = nan_logsumexp(log_r, axis=3, all_nan_return=np.nan)  # shape: (n_individuals, n_days, n_clusters)
    proba_this_cluster = np.nanmean(proba_this_cluster_per_day, axis=1) # shape: (n_individuals, n_clusters)
    chosen_cluster = np.argmax(proba_this_cluster, axis=1)  #  shape: (n_individuals,)

    n_ind_var = len(variable_names)
    ncol = 1  #np.ceil(np.sqrt(n_clusters)).astype(int)
    nrow = np.ceil(n_clusters/ncol).astype(int)
    plt.figure(figsize=(n_clusters*3,3))
    for k in range(n_clusters):
        plt.subplot(ncol, nrow, k+1)
        these_inds = (chosen_cluster == k)

        hist_array = np.zeros(n_ind_var)
        for i_var, vname in enumerate(variable_names):
            hist_array[i_var] = (ind_variables[these_inds,0,i_var] > 0).sum()
        prop_array = hist_array/ np.sum(hist_array)
        # plt.step(np.arange(n_ind_var), prop_array, c=colors_per_cluster[k,:], label=f'cluster {k} ({these_inds.sum()} stations)')
        # plt.bar(x=np.arange(n_ind_var), width=1,
        #         bottom=np.zeros(n_ind_var), height=prop_array,
        #         edgecolor=colors_per_cluster[k,:], facecolor=[0, 0, 0, 0], # alpha=0
                # label=f'cluster {k+1} ({these_inds.sum()} stations)', lw=2)
        plt.bar(x=np.arange(n_ind_var), width=1,
                bottom=np.zeros(n_ind_var), height=hist_array,
                facecolor=colors_per_cluster[k,:], edgecolor='k')
        if k % ncol == ncol-1:
            plt.xticks(np.arange(n_ind_var), variable_names, ha='left', rotation=-45)
        else:
            plt.xticks(np.arange(n_ind_var), [""]*n_ind_var)
        if k % nrow == 0: plt.ylabel("Number of samples")
        plt.grid(True)
        # plt.legend()



if __name__ =="__main__":

    # old_ind_array = ind_array.copy()

    chosen_variables =  ["strike", "lockdown_different", "school_holiday", "national_holiday",
                         "year_splines_d2_n21", "day_of_week", "rail_type"]

    data, variable_names, id_names = load_data("rail", ticket_types="merge", sampling=None,
                                                chosen_variables=chosen_variables, normalisation="none",
                                                centering="individual", scale='log10')

    (X_train, ind_array, day_array, variables_train) = data[0]


    # assert (ind_array == old_ind_array).all()

    plot_proportion_of_each(log_r_train, variables_train[0], variable_names,
                            colors_per_cluster=base_colors_per_cluster[:model.n_clusters,:])




