import numpy as np
import matplotlib.pyplot as plt




def plot_cluster_agreement(model, X, variables):
    """
    Compute the expectation, for all samples in some origin segment and cluster, of the
    log-likelihood of this element to belong in each other destination cluster.
    This only takes into account the Gaussian and exogenous variables, and ignores the
    mixture weights and segment locations.

    Parameters
    ----------
    model: an instance of GMM_segmentation
    X: np array with shape (n_individuals, n_days, dim)
    variables: triple of arrays


    Returns
    -------
    None

    """
    n_individuals, n_days, _ = X.shape
    mean_LL = np.zeros((model.n_clusters*model.n_segments, model.n_clusters*model.n_segments))
    std_LL  = np.zeros((model.n_clusters*model.n_segments, model.n_clusters*model.n_segments))
        # to compute the variance

    r = np.exp(model.E_step(X, variables)[0])  # shape: n_individuals, n_days, n_clusters, n_segments)
    (ind_variables, temp_variables, mixed_variables) = variables
    # if include_segment_ranges: t = np.linspace(0,1, n_days)

    for k_d in range(model.n_clusters):  # d = destination
        for s_d in range(model.n_segments):
            # if include_segment_ranges:
            #     this_u, this_v = model.u[k_d,s_d].reshape(1,-1), model.v[k_d,s_d].reshape(1,-1) # shape: (1,)
            #     log_kappa = this_u * t + this_v                      # log_kappa.shape: (n_days)
            #     log_kappa -= logsumexp(log_kappa)

            mu_ks = model.m[:,k_d,s_d][np.newaxis, np.newaxis,:]  +\
                (ind_variables   @ model.alpha[:,:,k_d,s_d])  +\
                (temp_variables  @  model.beta[:,:,k_d,s_d])  +\
                (mixed_variables @ model.gamma[:,:,k_d,s_d])  # shape: (n_individuals, n_days, dim)
            sigma_ks = model.sigma if model.covariance_type == 'tied' else  model.sigma[...,k_d,s_d]
            log_N = model.N(X, mu_ks, sigma_ks)  # shape: (n_individuals, n_days)

            for k_o in range(model.n_clusters):  # o = origin ~= prior
                for s_o in range(model.n_segments):
                    cluster_weight_o = r[:,:,k_o,s_o] / (np.nansum(r[:,:,k_o,s_o]) + 1e-30)  # shape: (n_individuals, n_days)

                    this_mean = np.nansum(cluster_weight_o * log_N) # the weights sum to one, so np.nansum(weights*A) is the mean of A
                    mean_LL[k_o * model.n_segments + s_o, k_d * model.n_segments + s_d] = this_mean

                    this_std = np.sqrt(np.nansum(cluster_weight_o * (log_N - this_mean)**2))
                    std_LL[k_o * model.n_segments + s_o, k_d * model.n_segments + s_d] = this_std

    LL_diagless = mean_LL.copy()
    for i in range(mean_LL.shape[0]): LL_diagless[i,i] = -np.inf
    max_LL = max(0, np.max(LL_diagless))
    chosen_percentile = 100*2/(model.n_clusters+4) if model.n_clusters >= 3 else 20 # empirical fitting
    min_LL = 10 * np.floor(np.percentile(mean_LL, chosen_percentile)/10)
    print("mean_LL", mean_LL)
    plt.figure(figsize=(15, 15))
    plt.imshow(mean_LL.T, vmin=min_LL, vmax=max_LL) # imshow swaps axes
    plt.colorbar()
    plt.xlabel("origin / explained")
    plt.ylabel("destination / explaining")
    xticklabels, yticklabels = [], []
    for k_d in range(model.n_clusters):
        for s_d in range(model.n_segments):
            if s_d == model.n_segments//2:
                xticklabels.append(f"s{s_d}\nc{k_d}")
                yticklabels.append(f"cluster {k_d}  segment {s_d}")
            else:
                xticklabels.append(f"s{s_d}")
                yticklabels.append(f"segment {s_d}")
            for k_o in range(model.n_clusters):
                for s_o in range(model.n_segments):
                    this_mean = mean_LL[k_o * model.n_segments + s_o, k_d * model.n_segments + s_d]
                    this_std =   std_LL[k_o * model.n_segments + s_o, k_d * model.n_segments + s_d]
                    LL_string = f"{this_mean:.2f} \n ±{this_std:.2f}"
                    # LL_string = f"o:{k_o},d:{k_d}"
                    textcolor = "k" if this_mean > (max_LL+min_LL)/2 else "w"
                    plt.text(x=(k_o*model.n_segments+s_o), y=(k_d*model.n_segments+s_d), # imshow swaps the y axis
                              s=LL_string, ha="center", va="center", c=textcolor, fontsize=6)


    plt.xticks(np.arange(len(xticklabels)), xticklabels)
    plt.yticks(np.arange(len(yticklabels)), yticklabels)





#%%
if __name__ == "__main__":
    from models.main_model import GMM_segmentation
    from visualizations.clustering_results import plot_mean_covar


    n_individuals, n_days = 100, 100

    model = GMM_segmentation(dim=2, n_clusters=2, n_segments=2, covariance_type="diag")

    colors = np.zeros((3,model.n_clusters, model.n_segments))

    # Cluster 0
    model.m[:,0,0] = np.array([0,0])
    model.sigma[:,0,0] = np.array([1,5])
    model.m[:,0,1] = np.array([30,0])
    model.sigma[:,0,1] = np.array([10,1])

    # Cluster 1
    model.m[:,1,0] = np.array([15,30])
    model.sigma[:,1,0] = np.array([2,2])
    model.m[:,1,1] = np.array([15,30])
    model.sigma[:,1,1] = np.array([15,5])

    colors[:,0,0] = np.array([0.00, 0.87, 0.43])
    colors[:,0,1] = np.array([0.65, 0.86, 0.07])
    colors[:,1,0] = np.array([0.71, 0.16, 0.38])
    colors[:,1,1] = np.array([0.56, 0.56, 0.87])


    chosen_clusters = np.floor(np.linspace(0, model.n_clusters, n_individuals, endpoint=False)).astype(int)
    chosen_segments = np.floor(np.linspace(0, model.n_segments, n_days,        endpoint=False)).astype(int)
    r = np.zeros((n_individuals, n_days, model.n_clusters, model.n_segments))
    X = np.zeros((n_individuals, n_days, 2))
    for i in range(n_individuals):
        c = chosen_clusters[i]

        for d in range(n_days):
            s = chosen_segments[d]
            r[i,d,c,s] = 1
            X[i,d,:] = model.m[:,c,s] + np.random.randn(2) * model.sigma[:,c,s]

    ind_variables   = np.zeros((n_individuals, 1,      0), dtype=float)
    temp_variables  = np.zeros((1,             n_days, 0), dtype=float)
    mixed_variables = np.zeros((n_individuals, n_days, 0), dtype=float)

    plt.figure(figsize=(10, 10))
    for c in range(model.n_clusters):
        individuals = (chosen_clusters == c)
        for s in range(model.n_segments):
            days = (chosen_segments == s)
            color = list(colors[:,c,s])

            plt.scatter(X[individuals,days,0].reshape(-1), X[individuals,days,1].reshape(-1), c=[color+[0.5]],  # add alpha
                        s=10, marker="d", label=f'cluster {c} segment {s}')
            plot_mean_covar(mu=model.m[:,c,s], cov=np.diag(model.sigma[:,c,s]), c=color)

    plt.legend()
    plot_cluster_agreement(model, X, (ind_variables, temp_variables, mixed_variables))




