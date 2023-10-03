import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import adjusted_rand_score

from models.main_model import GMM_segmentation
from experiments.compare_performance_real_data import evaluation_process

from visualizations.clustering_results import plot_cluster_ind_day, plot_model_mean, generate_colors
from visualizations.cluster_explanation import compute_plot_explanations
from visualizations.clustering_results_IdFM import plot_model_mean_unidimensional


np.random.seed(0)


# these are global variables (I know)
n_clusters = 4
n_segments = 4
n_ind_variables   = 1
n_temp_variables  = 1
n_mixed_variables = 1
dim = 1

AR_parameter = 0.9  # to generate the autoregressive dataset (we use AR(1))




def generate_synthetic_dataset(distribution, n_individuals, n_days, variable_contrib):
    """
    Parameters
    ----------
    distribution: string, one of "normal", "autoregressive", "lognormal"
        Changes the observation model
    n_individuals: int
    n_days: int
    variable_contrib: float
        the average norm for the contributions of the exogenous variables

    Returns
    -------
    X: np array with shape (n_individuals, n_days, dim)
    variables: triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables)
    r_gt: np array with shape (n_individuals, n_days, n_clusters, n_segments)
    params: a list of parameters: in order
        pi, u, v, m, alpha, beta, gamma, sigma
    """
    min_slope = 4.4 * (n_days * 0.1)  # from 10% to 90% (4.4) in less than a tenth of the interval

    assert (distribution in ["normal", "autoregressive", "lognormal"]), f"Unknown distribution '{distribution}'. Accepted values are 'normal', 'autoregressive', 'lognormal', 'real_IdFM'"

    t = np.linspace(0, 1, n_days)
    # pi = np.ones(n_clusters) / np.ones(n_clusters)
    pi = np.random.dirichlet(alpha = np.ones(n_clusters)*2)  # shape: (n_clusters,)

    segment_proportions = np.random.dirichlet(alpha = np.ones(n_segments)*2, size=n_clusters)
                # shape: (n_clusters, n_segments), sums to one along segments
    segment_borders = np.cumsum(segment_proportions, axis=1)   # shape: (n_clusters, n_segments), all values in [0,1]



    u = min_slope * np.arange(n_segments)[np.newaxis,:]  # shape: (1, n_segments)
    v = np.zeros((n_clusters, n_segments))
    v[:,1:] = np.cumsum(-np.diff(u, axis=1) * segment_borders[:,:-1], axis=1)

    u -= u.mean(axis=1, keepdims=True)
    v -= v.mean(axis=1, keepdims=True)

         # cluster assignment
    cluster_dummy = np.random.multinomial(1, pvals=pi, size=(n_individuals,))  # shape: (n_individuals, n_clusters)
    cluster_assignment = np.argmax(cluster_dummy, axis=1)                      # shape: (n_individuals,)
    r = np.zeros((n_individuals, n_days, n_clusters, n_segments))
    for k in range(n_clusters):

        selected_individuals = (cluster_assignment == k)

        probabilities = softmax(u*t[:,np.newaxis] + v[k:k+1,:], axis=1)  # shape: (n_days, n_segments)

        random_uniform_per_segment = np.random.uniform(0,1, size=(selected_individuals.sum(), n_days, n_segments))  # shape: (n_selected_ind, n_days, n_segments)
        random_weighted_per_segment = random_uniform_per_segment * probabilities[np.newaxis,:,:]                    # shape: (n_selected_ind, n_days, n_segments)
        segment_assignment = np.argmax(random_weighted_per_segment, axis=2)   # shape: (n_selected_ind, n_days)

        segment_index = np.arange(n_segments).reshape(1, 1, -1)  # shape: (1, 1, n_segments)
        r[selected_individuals,:,k,:] = (segment_assignment[:,:,np.newaxis] == segment_index)

    assert (np.sum(r, axis=(2,3)) == 1).all()  # all (ind, days) belong to exactly one segment from one cluster


    # Exogenous variables

    # We make no hypotheses on the distribution of exogenous variables so we may choose what we want
    ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
    temp_variables  = np.random.randn(1,             n_days, n_temp_variables)
    mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)

    # ind_variables  *= variable_contrib /  ind_variables.std(axis=0, keepdims=True)
    # temp_variables *= variable_contrib / temp_variables.std(axis=0, keepdims=True)

    #  Same argument for the contributions of the variables
    alpha = np.random.randn(n_ind_variables,   dim, n_clusters, n_segments) * variable_contrib
    beta  = np.random.randn(n_temp_variables,  dim, n_clusters, n_segments) * variable_contrib
    gamma = np.random.randn(n_mixed_variables, dim, n_clusters, n_segments) * variable_contrib

    sigma = np.ones((dim, n_clusters, n_segments)) * 1

    X = np.zeros((n_individuals, n_days, dim))
    if distribution == "normal":
        m = np.random.randn(dim, n_clusters, n_segments) * 1

        for k in range(n_clusters):
            for s in range(n_segments):
                this_cluster_mean  = m[:,k,s].reshape(1,1,-1) +\
                                     (ind_variables   @ alpha[:,:,k,s]) +\
                                     (temp_variables  @ beta[ :,:,k,s]) +\
                                     (mixed_variables @ gamma[:,:,k,s])
                noise = np.random.randn(*this_cluster_mean.shape) * np.sqrt(sigma[:,k,s])[np.newaxis,np.newaxis,:]
                this_cluster_realization = this_cluster_mean + noise
                X += r[:,:,k,s][:,:,np.newaxis] * this_cluster_realization

    elif distribution == "autoregressive":
        m = np.random.randn(dim, n_clusters, n_segments) * 1

        for k in range(n_clusters):
            for s in range(n_segments):
                this_cluster_center = np.zeros((n_days, dim))
                this_cluster_center[0,:] = np.random.randn(dim) * sigma[:,k,s]
                for t in range(1, n_days):
                    this_cluster_center[t,:] = AR_parameter * this_cluster_center[t-1,:]  +\
                                                np.random.randn(dim) * sigma[:,k,s]
                this_cluster_realization = this_cluster_center[np.newaxis,:,:]   +\
                                                 (ind_variables   @ alpha[:,:,k,s]) +\
                                                 (temp_variables  @ beta[ :,:,k,s]) +\
                                                 (mixed_variables @ gamma[:,:,k,s]) # shape: (n_individuals, n_days, dim)
                X += r[:,:,k,s][:,:,np.newaxis] * this_cluster_realization


    elif distribution == "lognormal":
        m = np.random.randn(dim, n_clusters, n_segments)

        for k in range(n_clusters):
            for s in range(n_segments):
                this_cluster_mean  = m[:,k,s].reshape(1,1,-1) +\
                                     (ind_variables   @ alpha[:,:,k,s]) +\
                                     (temp_variables  @ beta[ :,:,k,s]) +\
                                     (mixed_variables @ gamma[:,:,k,s])

                noise_normal = np.random.randn(*this_cluster_mean.shape) * np.sqrt(sigma[:,k,s])[np.newaxis,np.newaxis,:]
                noise_lognormal = np.exp(noise_normal)

                this_cluster_realization = this_cluster_mean + noise_lognormal
                X += r[:,:,k,s][:,:,np.newaxis] * this_cluster_realization


    params = [pi, u, v, m, alpha, beta, gamma, sigma]
    return X, (ind_variables, temp_variables, mixed_variables), r, params



#%%


def get_ari_from_r(r_gt, r_pred):
    """
    Parameters
    ----------
    r_gt:   np array with shape (n_individuals, n_days, n_clusters, n_segments)
    r_pred: np array with shape (n_individuals, n_days, n_clusters, n_segments)
        Note that replacing the respondibilities by their log works too, because
        we take the argmax.

    Returns
    -------
    ari: float, between -0.5 and 1
        (zero designates random assignment)
    """
    n_individuals, n_days,  n_clusters, n_segments = r_gt.shape
    r_gt_reshaped = r_gt.reshape(      n_individuals, n_days,  n_clusters * n_segments)
    r_gt_reshaped = r_gt_reshaped.reshape(n_individuals * n_days, n_clusters * n_segments)
    cluster_segment_gt = np.argmax(r_gt_reshaped, axis=1)  # shape: (n_individuals * n_days)

    r_pred_reshaped = r_pred.reshape(         n_individuals, n_days,  n_clusters * n_segments)
    r_pred_reshaped = r_pred_reshaped.reshape(n_individuals * n_days, n_clusters * n_segments)
    cluster_segment_pred = np.argmax(r_pred_reshaped, axis=1)  # shape: (n_individuals * n_days)

    return  adjusted_rand_score(cluster_segment_gt, cluster_segment_pred)








#%%

print('\n\n')
print("="*80)
print("\t\t\t contribution of the exogenous variables fixed to 1.")
print("="*80)


distribution_data = "normal"
variable_contrib = 1.
for n_individuals, n_days  in [(50,50), (100,100), (500,500), (1000,1000)]:


    results = {"Clust+Seg":[],
               "Reg > Clust > Seg":[],
               "(Clust+Reg) > (Seg+Reg)":[],
               "Reg > Clust+Seg":[],
               "Clust+Seg+Reg":[] }

    for i_trial in range(10):

        min_slope = 4.4 * (n_days * 0.1)  # from 10% to 90% (4.4) in less than a tenth of the interval
        X, variables, r_gt, _ = generate_synthetic_dataset(distribution_data, n_individuals, n_days, variable_contrib)
        def evaluation_fn_synthetic(model, X, variables, log_r):
            return get_ari_from_r(r_gt, log_r)  # we use neither the model nor X

        kwargs_model = {"covariance_type":"diag", "min_slope":min_slope}
        # here, the validation set equals the train set - because we can compare a (training) partition to a ground truth
        this_ari_dict, _ = evaluation_process(X, variables, X, variables, evaluation_fn=evaluation_fn_synthetic,
                                                     **kwargs_model)


        for k in results.keys():
            results[k].append(this_ari_dict[k])
    print('\n\n')
    print(f"I={n_individuals}, T={n_days}")
    for k, v in results.items():
        print(f"{k:25s}: ARI = {np.mean(v):.3f} +/- {np.std(v):.3f}")
    print('\n\n')





#%%

print('\n\n')
print("="*80)
print("\t\t\t with 100 individuals and as many days")
print("="*80)

distribution_data = "normal"
n_individuals, n_days = 100, 100
for variable_contrib in [0., 0.5, 1., 1.5]:

    results = {"Clust+Seg":[],
               "Reg > Clust > Seg":[],  # purely sequential
               "(Clust+Reg) > (Seg+Reg)":[],
               "Reg > Clust+Seg":[],
               "Clust+Seg+Reg":[] }

    for i_trial in range(10):

        min_slope = 4.4 * (n_days * 0.1)  # from 10% to 90% (4.4) in less than a tenth of the interval
        X, variables, r_gt, _ = generate_synthetic_dataset(distribution_data, n_individuals, n_days, variable_contrib)
        def evaluation_fn_synthetic(model, X, variables, log_r):
            return get_ari_from_r(r_gt, log_r)  # we use neither the model nor X

        kwargs_model = {"covariance_type":"diag", "min_slope":min_slope}
        this_ari_dict, _ = evaluation_process(X, variables, X, variables, evaluation_fn=evaluation_fn_synthetic,
                                                     **kwargs_model)


        for k in results.keys():
            results[k].append(this_ari_dict[k])


    print('\n\n')
    print(f"variable_contrib={variable_contrib}")
    for k, v in results.items():
        print(f"{k:25s}: ARI = {np.mean(v):.3f} +/- {np.std(v):.3f}")
    print('\n\n')


#%%



print('\n\n')
print("="*80)
print("\t   contribution of the exogenous variables fixed to 1 AND I = T = 100")
print("="*80)
variable_contrib = 1


n_individuals, n_days = 100, 100
for distribution_data in ["normal", "autoregressive", "lognormal"]:

    results = {"Clust+Seg":[],
               "Reg > Clust > Seg":[],  # purely sequential
               "(Clust+Reg) > (Seg+Reg)":[],
               "Reg > Clust+Seg":[],
               "Clust+Seg+Reg":[] }

    for i_trial in range(10):

        min_slope = 4.4 * (n_days * 0.1)  # from 10% to 90% (4.4) in less than a tenth of the interval
        X, variables, r_gt, _ = generate_synthetic_dataset(distribution_data, n_individuals, n_days, variable_contrib)
        def evaluation_fn_synthetic(model, X, variables, log_r):
            return get_ari_from_r(r_gt, log_r)  # we use neither the model nor X

        kwargs_model = {"covariance_type":"diag", "min_slope":min_slope}
        this_ari_dict, _ = evaluation_process(X, variables, X, variables, evaluation_fn=evaluation_fn_synthetic,
                                                     **kwargs_model)


        for k in results.keys():
            results[k].append(this_ari_dict[k])




        print('\n\n')
        print(f"distribution_data = {distribution_data}")
        for k, v in results.items():
            print(f"{k:25s}: ARI = {np.mean(v):.3f} +/- {np.std(v):.3f}")
        print('\n\n')