import numpy as np
from scipy.stats import t

from utils.nan_operators import nan_logsumexp





def compare_coefs(X1, X2, Y2, Y1, coefs_1, coefs_2, n_ddof=None, confidence_interval=None):
    """
    Parameters
    ----------
    X1: np array with shape (n_samples_1, n_dims)
    X2: np array with shape (n_samples_2, n_dims)
    Y1: np array with shape (n_samples_1,)
    Y2: np array with shape (n_samples_2,)
    coefs_1: np array with shape (n_dims,)
    coefs_2: np array with shape (n_dims,)
    n_ddof: int, optional.
        Degrees of freedom
    confidence_interval: float between 0 and 1, optional
        the degree of confidenfe for the interval. If omitted, no interval is returned

    Returns
    -------
    p_values: np array of floats  with shape (n_dims,)
    confidence_interval:  np array of floats with shape (n_dims,2)
        Returned only if confidence_interval has been specified
    """

    n_variables = coefs_1.shape[0]
    X = np.concatenate([X1, X2], axis=0)
    Y = np.concatenate([Y1, Y2], axis=0)
    if n_ddof is None: n_ddof =  X.shape[0] * (X1.shape[0] + X2.shape[0] - n_variables)
    intervals = np.zeros((n_variables, 2))


    p_values = np.zeros(n_variables)
    for i_v in range(n_variables):
        delta_coefs = coefs_2 - coefs_1
        delta_coefs_hypothesis = delta_coefs.copy()
        delta_coefs_hypothesis[i_v] = 0

        residuals = Y - (X @ delta_coefs)

        loc = delta_coefs_hypothesis[i_v] - delta_coefs[i_v]
        scale =  np.sqrt((1/n_ddof) *  np.nansum(residuals**2)) / np.sqrt(np.nansum((X[:,i_v] - np.mean(X[:,i_v]))**2) + 1e-15)

        t_statistic = loc / scale


        p_values[i_v] = 2 * min(t.cdf(-t_statistic, df=n_ddof),
                                t.cdf( t_statistic, df=n_ddof))

        if confidence_interval != None:
            intervals[i_v] = t.interval(confidence_interval, df=n_ddof, loc=loc, scale = scale)

    if confidence_interval is None:
        return p_values
    else:
        return p_values, intervals





if __name__ == "__main__":
    alpha = 0.05

    n_samples = 10000 # samples per test
    n_tests = 100
    n_variables = 3

    X1 = np.random.randn(n_tests, n_samples, n_variables)  # mean zero, cov 1
    X2 = np.random.randn(n_tests, n_samples, n_variables)  # mean zero, cov 1
    X1[:,:,1] *= 0
    X2[:,:,1] *= 1e-1
    coefs_1 = np.array([1, -3, 0.6]).reshape([-1,1])
    coefs_2 = np.array([1, -1, 1  ]).reshape([-1,1])
    Y1 = X1 @ coefs_1
    Y2 = X2 @ coefs_2
    Y1 += np.random.randn(*Y1.shape)
    Y2 += np.random.randn(*Y2.shape)

    avg_p_values = np.zeros(n_variables)
    for i_t in range(n_tests):
        p_values = compare_coefs(X1[i_t,...], X2[i_t,...], Y1[i_t,...], Y2[i_t,...], coefs_1, coefs_2)
        avg_p_values += (p_values)* (1/n_tests)


    print("average p-value:")
    print(avg_p_values)







def compare_means(Y1, Y2, n_ddof=None, confidence_interval=None):
    """
    Parameters
    ----------
    Y1: np array with shape (n_samples_1,)
    Y2: np array with shape (n_samples_2,)
    n_ddof: int, optional.
        Degrees of freedom
    confidence_interval: float between 0 and 1, optional
        the degree of confidenfe for the interval. If omitted, no interval is returned

    Returns
    -------
    p_value: float
    confidence_interval: couple of floats
        Returned only if confidence_interval has been specified
    """



    n1, n2 = Y1.shape[0], Y2.shape[0]
    if n_ddof is None: n_ddof = n1 + n2 -2

    if n1 == 0:
        n1 = 1
        Y1 = np.zeros(1)
    elif n2 == 0:
        n2 = 1
        Y2 = np.zeros(1)

    s = np.sqrt( ( (n1-1) * np.nanstd(Y1)**2 + (n2-1) * np.nanstd(Y2)**2 )/  n_ddof )
    scale =  (s * np.sqrt((1/n1) + (1/n2)))
    loc = (np.nanmean(Y1) - np.nanmean(Y2))

    t_statistic = loc / scale
    p_value = 2 * min(t.cdf(-t_statistic, df=n_ddof),
                      t.cdf( t_statistic, df=n_ddof))


    if confidence_interval is None:
        return p_value
    else:
        interval = t.interval(confidence_interval, df=n_ddof, loc=loc, scale = scale)
        return p_value, interval





if __name__ == "__main__":
    alpha = 0.05
    n_samples = 10000 # samples per test
    n_tests = 100
    n_variables = 3

    Y1 = np.random.randn(n_tests, n_samples)
    Y2 = np.random.randn(n_tests, n_samples) + 0.05

    proportion_significant = 0
    for i_t in range(n_tests):
        p_value = compare_means(Y1[i_t,:], Y2[i_t,:])
        proportion_significant += (p_value > alpha) * (1/n_tests)

    print(f"alpha = {alpha}")
    print("Proportion of tests finding significant differences:", proportion_significant)

#%%




def get_significant_differences(Y, variables, log_r, model, variable_names, alpha,
                                  print_vars=False, min_n_chosen=5, correction=None,
                                  variables_to_ignore=[], clusters_to_compare="all"):
    """
    Parameters
    ----------
    Y: np array with shape (n_individuals, n_days, dim)
    variables: triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables)
    log_r: p array with shape (n_individuals, n_days, dim)
    model:  a GMM_segmetnation instance. We need it for its alpha, beta, and gamma attributes.
    variable_names:
    alpha: float, between zero and one.
        The
    print_vars: bool, defaults to False
        Whether to print the results
    min_n_chosen: int, oprtional.
        Minimal nbumber of variables shown (even if the significance is low).
        Defaults to 5.
    correction: equal to either "Bonferroni" or None (default)
        How to account for the fact we test several hypotheses.
    variables_to_ignore: list, optional.
        Names of the variables that will not be printed (they will be in the output dictionnaries).
        Default: []
    clusters_to_compare: string or list, optional.
        If it is a list, it must be a list of couple of ints (each denoting a cluster number).
            The order of the elements of the couple does not matter.
        If it is a string, it must be eiyther "all", "none", or "following" (in which case
             we compare clusters 1 with 2, clusters 2 with 3, etc.)
        The default is "all".


    Returns
    -------
    Tuple of (nonzeros_variables, difference_clusters, difference_segments)

    nonzeros_variables: dict with
        keys = ints (clusters numbers)
        values = dict with
            keys = ints (segment indices)
            values = list of variable names (strings)

    difference_clusters: dict with
        keys = couple of ints (cluster indices)
        values = dict with
            keys = ints (segment numbers)
            values = list of variable names (strings)
        For example, difference_clusters[(3,4)][0] contains all the variables that differ significantly
        between cluster three and four in their first segment.

    difference_segments: dict with
        keys = ints (clusters numbers)
        values = dict with
            keys = couple of ints (segment indices)
            values = list of variable names (strings)
        For example, difference_segments[0][(2, 4)] contains all the variables that differ significantly
        between segments 2 and 4 of cluster 0.

    Notes:
    - The elements in the list are sorted by significanceance (i.e. by increasing p-value)
    - When the key is a couple, the clusters and segmetns are ordered:
         (0,4) is a possible key, (4,0) is not.
    - In all lists, the mean may also appear like a variable, denoted 'mean'
    """


    n_individuals, n_days, n_clusters, n_segments = log_r.shape
    if Y.shape[2] > 1: raise NotImplementedError("statistical tests only work if the explained variable (Y) is one-dimensional")

    all_variable_names = variable_names["individual"] + variable_names["temporal"] + variable_names["mixed"] + ["mean"]
    n_variables = model.n_ind_variables + model.n_temp_variables + model.n_mixed_variables + 1
    all_coefs = np.concatenate([model.alpha[:,0,:,:], model.beta[:,0,:,:], model.gamma[:,0,:,:]], axis=0)   # shape: (n_variables, n_clusters, n_segments)
    all_coefs = np.concatenate([all_coefs, model.m.reshape(1, n_clusters, n_segments)], axis=0)


    #  Remove the variables we want to ignore
    kept_indices = np.array([i for i in range(n_variables) if all_variable_names[i] not in variables_to_ignore])
    kept_mask = np.isin(np.arange(n_variables), kept_indices)
    n_kept_var = kept_mask.sum()
    kept_variable_names = [all_variable_names[i] for i in kept_indices]


    if correction is None:
        corrected_alpha = alpha
    elif correction == "Bonferroni":
        corrected_alpha = alpha / (1 + n_kept_var)  #  we divide by the number of hypotheses (we add one for the mean)

    proba_cluster_per_ind_day = nan_logsumexp(log_r, axis=3, all_nan_return=np.nan)
    log_rho_ik = np.nansum(proba_cluster_per_ind_day, axis=1)   # size: (n_individuals, n_clusters)
    selected_cluster = np.argmax(log_rho_ik, axis=1)  # shape; (n_individuals)


    def select_from_cluster_segment(k,s):
        these_inds = (selected_cluster == k)
        n_ind_this_k = these_inds.sum()
        this_k_Y = Y[these_inds,:,0]      # shape: (n_ind_this_k, n_days)
        this_k_Y = this_k_Y.reshape(-1)   # shape: (n_ind_this_k * n_days)

        log_r_segment = log_r[these_inds,:,k,:]  # shape: (n_individuals, n_days, n_segments)
        log_r_segment = log_r_segment.copy()
        log_r_segment[np.isnan(log_r_segment)] = -np.inf # np.argmax considers nan is +inf
        chosen_s = np.argmax(log_r_segment, axis=2).reshape(-1)  # shape: (n_individuals, n_days)

        this_cluster_N = (these_inds.sum()) * n_days
        ones = np.ones((these_inds.sum(), n_days, 1)) # for easier broadcast
        all_variables = np.concatenate([ (variables[0][these_inds,:,:] * ones).reshape(this_cluster_N, model.n_ind_variables),
                                         (variables[1]                 * ones).reshape(this_cluster_N, model.n_temp_variables),
                                          variables[2][these_inds,:,:].reshape(        this_cluster_N, model.n_mixed_variables),
                                          ones.reshape(this_cluster_N, 1)],   # this one is the mean
                                       axis=1)  # shape: (n_ind_this_k * n_days, n_variables)
        # all_variables = all_variables[:,kept_mask]

        segment_ind_days = (chosen_s == s).reshape(-1)

        this_ks_Y = this_k_Y[segment_ind_days].reshape(-1)
        this_ks_variables = all_variables[segment_ind_days,:]
        return this_ks_Y, this_ks_variables, n_ind_this_k




    if print_vars:
        print("\n"*3)
        print('='*80)
        print("\t\t significant variables")
        print('='*80)

    nonzeros_variables = {}
    for k in range(n_clusters):
        nonzeros_variables[k] = {}
        these_inds = (selected_cluster == k)
        n_ind_this_k = these_inds.sum()
        if print_vars: print(f"\n\nCluster {k+1} ({n_ind_this_k} individuals):")
        for s in range(n_segments):
            Y1, variables_1, n_ind_1 = select_from_cluster_segment(k,s)
            Y2, variables_2, n_ind_2 = np.zeros(0), np.zeros((0, n_variables)), 0
            n_samples_ddof = Y1.shape[0]

            p_values = compare_coefs(variables_1, variables_2, Y1, Y2,
                                     all_coefs[:,k,s], all_coefs[:,k,s] * 0,
                                      n_ddof = n_samples_ddof - n_variables)
            p_values[-1] = compare_means(Y1, Y2, n_ddof=n_samples_ddof - n_variables)# we treat the mean separately

            # sort by p-value
            order_p_values = np.argsort(p_values) # sort ascending
            nonzeros_variables[k][s] = [all_variable_names[i] for i in order_p_values if p_values[i] < corrected_alpha ]

            if print_vars:
                kept_p_values = p_values[kept_mask]
                order = np.argsort(kept_p_values)
                print(f"\tSegment {s+1}")
                for i_chosen in range(n_kept_var):
                    if (kept_p_values[order[i_chosen]] < corrected_alpha) or (i_chosen < min_n_chosen):
                        value = all_coefs[kept_mask,k,s][order[i_chosen]]
                        print(f"\t\t{i_chosen+1}) {kept_variable_names[order[i_chosen]]:25s} (p={kept_p_values[order[i_chosen]]:.3e}, {value:.3f}) ")







    if print_vars:
        print("\n"*3)
        print('='*80)
        print("\t\t Difference between segments")
        print('='*80)

    difference_segments = {}
    for k in range(n_clusters):
        these_inds = (selected_cluster == k)
        n_ind_this_k = these_inds.sum()
        difference_segments[k] = {}
        if print_vars:  print(f"\n\nCluster {k+1} ({n_ind_this_k} individuals):")
        for s1 in range(n_segments-1):
            Y1, variables_1, n_ind_1 = select_from_cluster_segment(k,s1)
            for s2 in range(s1+1,n_segments):
                Y2, variables_2, n_ind_2 = select_from_cluster_segment(k,s2)
                n_samples_ddof = Y1.shape[0] + Y2.shape[0]

                p_values = compare_coefs(variables_1, variables_2, Y1, Y2,
                                         all_coefs[:,k,s1], all_coefs[:,k,s2],
                                          n_ddof = n_samples_ddof - n_variables)
                p_values[-1] = compare_means(Y1, Y2, n_ddof=n_samples_ddof - n_variables)# we treat the mean separately

                order_p_values = np.argsort(p_values)
                difference_segments[k][(s1, s2)] = [all_variable_names[i] for i in order_p_values if p_values[i] < corrected_alpha ]

                if print_vars:
                    kept_p_values = p_values[kept_mask]
                    order = np.argsort(kept_p_values)
                    print(f"\tbetween segment {s1+1} and {s2+1}")
                    for i_chosen in range(n_kept_var):
                        if (kept_p_values[order[i_chosen]] < corrected_alpha) or (i_chosen < min_n_chosen):
                            value_s1 = all_coefs[kept_mask,k,s1][order[i_chosen]]
                            value_s2 = all_coefs[kept_mask,k,s2][order[i_chosen]]
                            print(f"\t\t{i_chosen+1}) {kept_variable_names[order[i_chosen]]:25s} (p={kept_p_values[order[i_chosen]]:.3e}, {value_s1:.3f} -> {value_s2:.3f})")






    if print_vars:
        print("\n"*3)
        print('='*80)
        print("\t\t Difference between clusters")
        print('='*80)
    difference_clusters = {}
    for s in range(n_segments):
        if print_vars: print(f"\n\nSegment {s+1}:")
        for (k1, k2) in clusters_to_compare:
            if k2 < k1: (k1, k2) = (k2, k1) # ordering

            Y1, variables_1, n_ind_1 = select_from_cluster_segment(k1,s)
            Y2, variables_2, n_ind_2 = select_from_cluster_segment(k2,s)
            if (k1, k2) not in difference_clusters:
                difference_clusters[(k1, k2)] = {}

            n_samples_ddof = Y1.shape[0] + Y2.shape[0]

            p_values = compare_coefs(variables_1, variables_2, Y1, Y2,
                                     all_coefs[:,k1,s], all_coefs[:,k2,s],
                                     n_ddof = n_samples_ddof - n_variables)
            p_values[-1] = compare_means(Y1, Y2, n_ddof=n_samples_ddof - n_variables)# we treat the mean separately

            order_p_values = np.argsort(p_values)
            difference_clusters[(k1, k2)][s] = [all_variable_names[i] for i in order_p_values if p_values[i] < corrected_alpha ]

            if print_vars:
                kept_p_values = p_values[kept_mask]
                order = np.argsort(kept_p_values)
                print(f"\tbetween cluster {k1+1} and {k2+1}")
                for i_chosen in range(n_kept_var):
                    if (kept_p_values[order[i_chosen]] < corrected_alpha) or (i_chosen < min_n_chosen):
                        value_k1 = all_coefs[kept_mask,k1,s][order[i_chosen]]
                        value_k2 = all_coefs[kept_mask,k2,s][order[i_chosen]]

                        print(f"\t\t{i_chosen+1}) {kept_variable_names[order[i_chosen]]:25s} (p={kept_p_values[order[i_chosen]]:.3e}, {value_k1:.3f} -> {value_k2:.3f})")


    return  (nonzeros_variables, difference_clusters, difference_segments)





