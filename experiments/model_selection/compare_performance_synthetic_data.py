"""
This file serves for model selection. It runs models with varying number of clusters and records the
resulting log-likelihood for another script to use, slope_heuristic.py.

We advise to run several instances of this script to use the several cores of your computer optimally.
In this case, each instance focuses on one number of clusters.
Each script writes a dummy file on disk to tell the others it started working on a
 specific numnber of cluster, and records the results in the .pickle once it is done.

As this script records everything on the disk, it is written to be launched with a large number of clusters
and interrupted to be restarted later on.
"""


from copy import deepcopy
import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


from preprocessing.IdFM.load_raw_data import IdFM_origin
from preprocessing.IdFM.load_data import load_data
from models.main_model import GMM_segmentation

use_synthetic_data = False


if use_synthetic_data:


    from utils.data_generation import generate_inconsistent_data  # inconsistent temporally

    n_clusters_GT = 4
    n_segments_GT = 3
    n_ind_variables   = 2
    n_temp_variables  = 2
    n_mixed_variables = 2
    n_individuals = 100
    n_days = 300
    dim = 1

    t = np.linspace(0, 1, n_days)
    min_slope = 4.4 * (n_days * 0.1)  # from 1% to 99% (4.4) in less than a tenth of the interval

    pi = np.random.dirichlet(alpha = np.ones(n_clusters_GT)*2)  # shape: (n_clusters,)
    m = np.random.randn( dim, n_clusters_GT, n_segments_GT)
    segment_proportions = np.random.dirichlet(alpha = np.ones(n_segments_GT)*2, size=n_clusters_GT)
                # shape: (n_clusters, n_segments), sums to one along segments
    segment_borders = np.cumsum(segment_proportions, axis=1)   # shape: (n_clusters, n_segments), all values in [0,1]

    u = min_slope * np.arange(n_segments_GT)[np.newaxis,:]  # shape: (1, n_segments)
    v = np.zeros((n_clusters_GT, n_segments_GT))
    v[:,1:] = np.cumsum(-np.diff(u, axis=1) * segment_borders[:,:-1], axis=1)

    u -= u.mean(axis=1, keepdims=True)
    v -= v.mean(axis=1, keepdims=True)

         # cluster assignment
    cluster_dummy = np.random.multinomial(1, pvals=pi, size=(n_individuals,))  # shape: (n_individuals, n_clusters)
    cluster_assignment = np.argmax(cluster_dummy, axis=1)                      # shape: (n_individuals,)
    r = np.zeros((n_individuals, n_days, n_clusters_GT, n_segments_GT))
    for k in range(n_clusters_GT):

        selected_individuals = (cluster_assignment == k)

        probabilities = softmax(u*t[:,np.newaxis] + v[k:k+1,:], axis=1)  # shape: (n_days, n_segments)

        random_uniform_per_segment = np.random.uniform(0,1, size=(selected_individuals.sum(), n_days, n_segments_GT))  # shape: (n_selected_ind, n_days, n_segments)
        random_weighted_per_segment = random_uniform_per_segment * probabilities[np.newaxis,:,:]                    # shape: (n_selected_ind, n_days, n_segments)
        segment_assignment = np.argmax(random_weighted_per_segment, axis=2)   # shape: (n_selected_ind, n_days)

        segment_index = np.arange(n_segments_GT).reshape(1, 1, -1)  # shape: (1, 1, n_segments)
        r[selected_individuals,:,k,:] = (segment_assignment[:,:,np.newaxis] == segment_index)

    assert (np.sum(r, axis=(2,3)) == 1).all()  # all (ind, days) belong to exactly one segment from one cluster


    # Exogenous variables

    # We make no hypotheses on the distribution of exogenous variables so we may choose what we want
    ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
    temp_variables  = np.random.randn(1,             n_days, n_temp_variables)
    mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)
    variables = (ind_variables, temp_variables, mixed_variables)

    #  Same argument for the contributions of the variables
    alpha = np.random.randn(n_ind_variables,   dim, n_clusters_GT, n_segments_GT)
    beta  = np.random.randn(n_temp_variables,  dim, n_clusters_GT, n_segments_GT)
    gamma = np.random.randn(n_mixed_variables, dim, n_clusters_GT, n_segments_GT)
    contributions = (alpha, beta, gamma)
    sigma = np.ones((dim,n_clusters_GT,n_segments_GT)) * 0.5**2


    X = generate_inconsistent_data(r, m, contributions, sigma, variables, covariance_type="diag")

    results = {}
    n_clusters_array = np.arange(1,9+1)
    n_segments_array = np.arange(1,9+1)
    n_init = 3

    LL_matrix  = np.zeros((len(n_clusters_array), len(n_segments_array), n_init))
    BIC_matrix = np.zeros((len(n_clusters_array), len(n_segments_array), n_init))
    ICL_matrix = np.zeros((len(n_clusters_array), len(n_segments_array), n_init))


    for i_clust, n_clusters in enumerate(n_clusters_array):
        for i_seg, n_segments in enumerate(n_segments_array):
            results[n_clusters] = []
            print(f"Benning with {n_clusters} n_cluster and {n_segments} segments")
            for i_init in range(n_init):

                model = GMM_segmentation(dim=1, n_clusters=n_clusters, n_segments=n_segments,
                                         n_variables=(v.shape[2] for v in variables),
                                         covariance_type="diag", min_slope=min_slope)


                model.EM(X, variables, init_with_CEM=False, init_method=('parameters', ("random", "uniform")),
                        CEM=False, print_every=0, plot_progression=False)

                LL =  model.LL_list[-1]
                BIC = model.BIC(X, variables)
                ICL = model.ICL(X, variables)
                results[n_clusters].append((deepcopy(model), LL, BIC, ICL))
                print(f'\t Initialization n°{i_init}: LL = {LL:.2f},  BIC = {BIC:.2f},  ICL = {ICL:.2f}')

                LL_matrix[ i_clust, i_seg, i_init] = LL
                BIC_matrix[i_clust, i_seg, i_init] = BIC
                ICL_matrix[i_clust, i_seg, i_init] = ICL



    metrics = {"LL":LL_matrix, "BIC":BIC_matrix, "ICL":ICL_matrix}
    for metric, matrix in metrics.items():
        plt.figure()
        matrix = matrix.copy()
        matrix[matrix == 0.] = np.nan
        plt.imshow(np.nanmean(matrix, axis=2))
        plt.xlabel("n_clusters")
        plt.ylabel("n_segments")
        plt.colorbar()
        plt.title(f"Data generated with {n_clusters_GT} clusters and {n_segments_GT} segments\nmetric='{metric}'")


#%%
else:
#%%

    results = {}  # triple (model, LL_train, LL_val)
    n_init = 3

    np.random.seed(0)
    n_clusters_array = np.arange(1, 20+1)


    chosen_variables =  ["strike", "strike_2019", "lockdown_different", "school_holiday", "national_holiday",
                          "year_splines_d2_n21", "day_of_week", "time"]

    data, variable_names, id_names, _ = load_data("rail", ticket_types="merge", sampling=None, start_date=datetime.date(2017,1,1),
                                                chosen_variables=chosen_variables, normalisation="individual",
                                                remove_mean="none", scale='log10')
    X, _, _, exogenous_variables = data[0]  # we do not train/test split herre to be able to use cross-validation
    ind_variables, temp_variables, mixed_variables = exogenous_variables
    n_individuals, n_days, _ = X.shape

    proportion_train = 1.
    n_train = int(n_individuals * proportion_train)

    for n_clusters in n_clusters_array:
        n_segments = 2



        if  f"results_{n_clusters}_clusters_{n_segments}_segments.pickle" not in os.listdir("results/n_clusters_selection/IdFM"):

            results[(n_clusters, n_segments)] = []
            print(f"\n\nBegin n_clusters = {n_clusters}, n_segments = {n_segments}")

            for i_init in range(n_init):

                individuals = np.arange(n_individuals)
                np.random.shuffle(individuals)
                val_individuals = np.zeros(n_individuals, dtype=bool)
                val_individuals[individuals > n_train] = True
                X_train = X[~val_individuals,...]
                X_val  =  X[ val_individuals,...]
                ind_var_train = ind_variables[~val_individuals,...]
                ind_var_val  =  ind_variables[ val_individuals,...]
                mixed_variables_train = mixed_variables[~val_individuals,...]
                mixed_variables_val   = mixed_variables[ val_individuals,...]
                variables_train = (ind_var_train, temp_variables, mixed_variables_train)
                variables_val   = (ind_var_val,   temp_variables, mixed_variables_val)


                model = GMM_segmentation(dim=1, n_clusters=n_clusters, n_segments=n_segments,
                                         n_variables=(v.shape[2] for v in variables_train),
                                         covariance_type="diag", min_slope=4.4 * n_days/(90))
                def newinit(*args, **kwargs):
                    GMM_segmentation.EM_init(model, *args, **kwargs)
                    print("init over")
                model.EM_init = newinit

                model.EM(X_train, variables_train,
                         init_with_CEM=False, init_method=('parameters', ("random", "uniform")),
                        CEM=False, print_every=0)

                LL_train = model.LL_list[-1]
                _, LL_val = model.E_step(X_val, variables_val)
                ICL_train = model.ICL(X_train, variables_train)
                this_result = (deepcopy(model), LL_train, ICL_train)
                results[(n_clusters, n_segments)].append(this_result)
                print(f'\t\t {n_clusters} clusters, Initianlzation n°{i_init}: LL_train={LL_train:.3e}, ICL_train={ICL_train:.3e}')

                with open(f"results/n_clusters_selection/IdFM/results_{n_clusters}_clusters_{n_segments}_segments.pickle", "wb") as f:
                    pickle.dump(results, f)



