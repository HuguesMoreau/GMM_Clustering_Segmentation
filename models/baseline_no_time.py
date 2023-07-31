"""
The GMM defined here takes into account the contributions of the exogenous variables,
but does create segments
i.e., it only compute clusters for the couples (individual, day), regardless of the
days' ordering

/!\ This model inherits from the Clutering - segmentation model which appears in the publication.
It has S = 1 segments.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from scipy.special import logsumexp


from utils.data_generation import generate_inconsistent_data  # for testing
from utils.exogenous_utils import explain_contributions_exogenous
from models.main_model import GMM_segmentation
from visualizations.clustering_results import generate_colors


#%%

class GMM_timeless(GMM_segmentation):
    def __init__(self, dim:int, n_clusters:int,
                 n_variables:tuple=(0, 0, 0),
                 covariance_type:str="spherical"):
        """
        Parameters
        ----------
        dim: int
        n_clusters: int
        n_individual_variables: int, optional (default value is 0)
        n_temporal_variables: int, optional (default value is 0)
        covariance_type: str, optional. Must be one of:
            'full': each component has its own general covariance matrix.
            'tied': all components share the same general covariance matrix.
            'diag': each component has its own diagonal covariance matrix.
            'spherical' (default): each component has its own single variance.
            (behaves similarly to the sklearn implementation)
        """
        n_segments=1
        GMM_segmentation.__init__(self, dim, n_clusters, n_segments,
                     n_variables, covariance_type)
        del self.u, self.v

        self.n_parameters = self.m.size + self.alpha.size + self.beta.size +\
                            self.n_sigma_parameters + self.pi.size - 1
                            # do not forget the -1, it is the most important term :)



    def E_step(self, X, variables):
        """
        Parameters
        ----------
        X: np array with shape (n_individuals, n_days, dim)
        ind_variables:  np array with shape (n_individuals, n_ind_variables)
        temp_variables: np array with shape (n_days,        n_temp_variables)

        Returns
        -------
        log_r: np array with shape (n_individuals, n_days, n_clusters, 1)
            the natural log of the probability of belonging to a cluster for a
            couple (individual, day)
        log_likelihood: scalar
            The computation of the log_likelihood is greatly simplified using
            some intermediary variables we compute here.

        """
        n_individuals, n_days, _ = X.shape
        ind_variables, temp_variables, mixed_variables = self.unpack_vars(variables, n_individuals, n_days)
        log_p = np.zeros((n_individuals, n_days, self.n_clusters,1))  # p = P(X|theta)

        for k in range(self.n_clusters):
            mu_k = self.m[:,k,0][np.newaxis, np.newaxis,:]  +\
                (ind_variables   @ self.alpha[:,:,k,0]) +\
                (temp_variables  @  self.beta[:,:,k,0]) +\
                (mixed_variables @ self.gamma[:,:,k,0])# shape: (n_individuals, n_days, dim)
            sigma = self.sigma if self.covariance_type == 'tied' else  self.sigma[...,k,0]
            log_p[:,:,k,0] = np.log(self.pi[k]) + self.N(X, mu_k, sigma)
        log_likelihood = np.sum(logsumexp(log_p, axis=(2,3)), axis=(0,1))
        log_p -= logsumexp(log_p, axis=(2,3), keepdims=True)
        log_r = log_p   # we should have done log_r = log_p - logsumexp(log_p,...)
            # but this would have meant usign twice as much memory

        return log_r, log_likelihood




    def M_step(self, X, variables, log_r):
        """
        Parameters
        ----------
        X: np array with shape (n_individuals, n_days, dim)
        ind_variables:  np array with shape (n_individuals, n_ind_variables)
        temp_variables: np array with shape (n_days,        n_temp_variables)
        log_r: np array with shape (n_individuals, n_days, n_clusters)
            the natural log of the probability of belonging to a cluster for a
            couple (individual, day)
        Returns
        -------
        None (replaces the values in the parameters with their values at the next iteration)
        """
        n_individuals, n_days, _ = X.shape
        ind_variables, temp_variables, mixed_variables = self.unpack_vars(variables, n_individuals, n_days)

        count = np.sum(np.exp(log_r), axis=(0,1)) + np.finfo(float).eps   # shape: (n_clusters, 1)
        count += 1/2 # Dirichlet prior with alpha = 1/2
        new_pi = count/ np.sum(count)                                     # shape: (n_clusters,1)


        variables = self.unpack_vars(variables)
        self.m, self.alpha, self.beta, self.gamma = explain_contributions_exogenous(X, variables, log_r)
        proportion_variance = np.zeros(self.n_clusters)

        # Computation of the covariance
        new_sigma = np.zeros_like(self.sigma)
        for k in range(self.n_clusters):
            normalized_log_r = log_r[:,:,k,0] - logsumexp(log_r[:,:,k,0], axis=(0,1))  # shape: (n_ind, n_days)
            this_cluster_weight = np.exp(normalized_log_r)[:,:,np.newaxis]  # shape: (n_ind, n_days, 1), sums to one
            assert np.isclose(this_cluster_weight.sum(), 1)

            X_bar = X.copy() - self.m[:,k,0].reshape(1,1,-1)
            var_per_axis = np.sum(this_cluster_weight * (X_bar)**2, axis=(0,1)) # shape: (dim,)
            total_variance = np.sum(var_per_axis)
            X_bar -= (ind_variables   @ self.alpha[:,:,k,0])
            X_bar -= (temp_variables  @ self.beta[ :,:,k,0])
            X_bar -= (mixed_variables @ self.gamma[:,:,k,0])
            variance_after_regression = np.sum(this_cluster_weight * (X_bar)**2)
            proportion_variance[k] = 1 - variance_after_regression/total_variance


            if self.covariance_type == "spherical":
                sigma_per_dimension = (this_cluster_weight * (X_bar**2)).sum(axis=(0,1))
                    # here, we use sum() instead of mean() because this_cluster_weight already sums to one
                    # it plays the role of the 1/n
                new_sigma[k,0] = np.mean(sigma_per_dimension) + np.finfo(float).eps  # adding an epsilon for stability
            elif self.covariance_type == "diag":
                sigma_per_dimension = (this_cluster_weight * (X_bar**2)).sum(axis=(0,1))
                new_sigma[:,k,0] = sigma_per_dimension + np.finfo(float).eps

            else:
                this_cov = np.tensordot(this_cluster_weight * X_bar, X_bar, axes=[[0,1],[0,1]])
                this_cov += np.finfo(float).eps  * np.eye(self.dim) # regularization
                if self.covariance_type == "full":
                    new_sigma[:,:,k,0] = this_cov

                elif self.covariance_type == "tied":
                    new_sigma += self.pi[k,0] * this_cov

            del X_bar

        self.pi = new_pi
        # self.m, self.alpha, self.beta = new_m, new_alpha, new_beta
        self.sigma = new_sigma
        self.linear_regression_variance_history.append(proportion_variance)







    def EM(self, X, variables, init_method, print_every=10):
        """
        Parameters
        ----------
        X: np array with shape (n_individuals, n_days, dim)
        ind_variables:  np array with shape (n_individuals, n_ind_variables)
        temp_variables: np array with shape (n_days,        n_temp_variables)
            The exogenous variables need at least to have the same variance, which will
            happen in practice because we will standard-scale them.
        init_method: string
            How to initialize the mean m. Must be one of:
            - "kmeans" where the couples (individual, day) play the role of samples, and hours are features
            - "random_data" for each cluster, draw a Gaussan where the mean and covariance are the ones of the provided sdataset X
            - "random_absolute" for each cluster, draw a Gaussan where the mean and covariance zero and the identity
            Note that the intitialization for the exogenous variables' contributions and covariance
            is not changed by this parameter
        print_every: positive int, optional
            EM prints the log likely hood every X iterations.
            If it is set to zero, the function prints nothing.
            Defaults to 10.

        Returns
        -------
        LL_list: list of (negative) floats
        """
        self.n_individuals, self.n_days, _ = X.shape
            # will be used to compute a BIC

            # initialization:
        self.EM_init(X, variables, init_method)

            # EM-loop
        self.LL_list = []
        best_LL  = -np.inf
        i = 0
        iter_without_increasing = 0
        while iter_without_increasing < 10 and i < 1e4 :
            log_r, current_LL = self.E_step(X, variables)
            self.LL_list.append(current_LL)
            self.M_step(X, variables, log_r)
            self.proportions_history.append(np.copy(self.pi))

            improvement = current_LL - best_LL > np.abs(best_LL * 1e-2) if i > 0 else True
            if improvement:
                best_LL = current_LL
                iter_without_increasing = 0
            else:
                iter_without_increasing += 1

            if print_every > 0:
                if i%print_every == 0: print(f'\titeration {i}, LL={current_LL:.3f}')
            i += 1

        return self.LL_list









if __name__ == "__main__":
    n_clusters = 5
    n_ind_variables  = 1
    n_temp_variables = 1
    n_mixed_variables = 1
    n_individuals = 50
    n_days = 50
    dim = 1

    base_colors = generate_colors(n_clusters, method="basic colors")

    r_gt = np.zeros((n_individuals, n_days, n_clusters, 1))
    clusters_gt = np.random.randint(0, n_clusters, size=(n_individuals, n_days))
    for i in range(r_gt.shape[0]):
        for j in range(r_gt.shape[1]):
            r_gt[i,j,clusters_gt[i,j],0] = 1

    m_gt =     np.random.randn(                   dim, n_clusters, 1) * 3
    alpha_gt = np.random.randn(n_ind_variables,   dim, n_clusters, 1) * 2
    beta_gt  = np.random.randn(n_temp_variables,  dim, n_clusters, 1) * 2
    gamma_gt = np.random.randn(n_mixed_variables, dim, n_clusters, 1) * 2
    sigma_gt = np.ones((n_clusters, 1)) * 0.15
    ind_variables   = np.random.randint(0, 2, (n_individuals, 1,      n_ind_variables))
    temp_variables  = np.random.randint(0, 2, (1,             n_days, n_temp_variables))
    mixed_variables = np.random.randint(0, 2, (n_individuals, n_days, n_mixed_variables))
    variables = (ind_variables, temp_variables, mixed_variables)

    contributions_gt = (alpha_gt, beta_gt, gamma_gt)
    X = generate_inconsistent_data(r_gt, m_gt, contributions_gt, sigma_gt, variables)
    # plot_2D_clustering(X, r_gt, ind_variables, temp_variables, title='ground truth',
    #                     colors_per_cluster=base_colors)

    model = GMM_timeless(dim, n_clusters, (v.shape[2] for v in variables))
    LL_list = model.EM(X, variables, init_method=('parameters', ("random","kmeans")))
    estimated_r = np.exp(model.E_step(X, variables)[0])
    # plot_2D_clustering(X, estimated_r, ind_variables, temp_variables, title='model $using$ the variables',
    #                     colors_per_cluster=base_colors)

    plt.figure(figsize=(7,7))
    plt.plot(LL_list)
    plt.title("Log_likelihood of the model $using$ the exogenous variables")
    plt.grid(True)
    variable_names = {"individual":[f"ind. variable {i}"  for i in range(n_ind_variables) ],
                      "temporal":  [f"temp. variable {i}" for i in range(n_temp_variables)]}



    model = GMM_timeless(dim, n_clusters, (0, 0, 0))
    empty_vars = (np.zeros((n_individuals, 1, 0)), np.zeros((1, n_days, 0)), np.zeros((n_individuals, n_days, 0)))
    LL_list = model.EM(X, empty_vars, init_method=('parameters', ("random","kmeans")))
    estimated_r = np.exp(model.E_step(X, empty_vars)[0])
    # plot_2D_clustering(X, estimated_r, ind_variables, temp_variables, title='model $ignoring$ the variables',
    #                     colors_per_cluster=base_colors)

    plt.figure(figsize=(7,7))
    plt.plot(LL_list)
    plt.title("Log_likelihood of the model $ignoring$ the exogenous variables")
    plt.grid(True)





#%%


    n_clusters = 5
    n_ind_variables  = 3
    n_temp_variables = 2
    n_mixed_variables = 1
    n_individuals = 100
    n_days = 50
    dim = 10

    def get_arand_score(X, variables, clusters_gt):
        n_clusters = clusters_gt.max()+1
        dim = X.shape[2]
        model = GMM_timeless(dim, n_clusters, [v.shape[2] for v in variables])
        model.EM(X, variables, init_method=('parameters', ("random","kmeans")), print_every=0)
        estimated_r, _ = model.E_step(X, variables)
        hard_cluster_assignment = np.argmax(estimated_r, axis=2)
        arand = adjusted_rand_score(clusters_gt.reshape(-1), hard_cluster_assignment.reshape(-1))
        return arand


    n_repetitions = 5
    sigmas_array = 10**(np.linspace(-2,1,10))
    all_delta_scores = np.zeros((sigmas_array.shape[0], n_repetitions, 2))
    for i_s, sigma in enumerate(sigmas_array):
        for i_repetition in range(n_repetitions):

            r_gt = np.zeros((n_individuals, n_days, n_clusters, 1))
            clusters_gt = np.random.randint(0, n_clusters, size=(n_individuals, n_days))
            for i in range(r_gt.shape[0]):
                for j in range(r_gt.shape[1]):
                    r_gt[i,j,clusters_gt[i,j], 0] = 1

                    # continuous exogenous variables in random directions
            m_gt =     np.random.randn(                   dim, n_clusters, 1)
            alpha_gt = np.random.randn(n_ind_variables,   dim, n_clusters, 1)
            beta_gt  = np.random.randn(n_temp_variables,  dim, n_clusters, 1)
            gamma_gt = np.random.randn(n_mixed_variables, dim, n_clusters, 1)
            sigma_gt = np.ones((n_clusters, 1)) * sigma

            ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
            temp_variables  = np.random.randn(1,             n_days, n_temp_variables)
            mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)
            variables = (ind_variables, temp_variables, mixed_variables)
            empty_vars = [v[:,:,2:1] for v in variables]


            X = generate_inconsistent_data(r_gt, m_gt, (alpha_gt, beta_gt, gamma_gt), sigma_gt, variables)
            arand_with_variables    = get_arand_score(X, variables,  clusters_gt)
            arand_without_variables = get_arand_score(X, empty_vars, clusters_gt)
            all_delta_scores[i_s, i_repetition,0] = arand_with_variables - arand_without_variables


                    # continuous exogenous variables in cluster directions
            for i_iv in range(n_ind_variables):
                cluster_from = np.arange(n_clusters)
                cluster_to   = np.random.randint(0, n_clusters, (n_clusters,))
                alpha_gt[i_iv,:,:] += m_gt[:,cluster_to] - m_gt[:,cluster_from]
            for i_tv in range(n_temp_variables):
                cluster_from = np.arange(n_clusters)
                cluster_to   = np.random.randint(0, n_clusters, (n_clusters,))
                beta_gt[ i_tv,:,:] += m_gt[:,cluster_to] - m_gt[:,cluster_from]

            X = generate_inconsistent_data(r_gt, m_gt, (alpha_gt, beta_gt, gamma_gt), sigma_gt, variables)
            arand_with_variables    = get_arand_score(X, variables,  clusters_gt)
            arand_without_variables = get_arand_score(X, empty_vars, clusters_gt)
            all_delta_scores[i_s, i_repetition,1] = arand_with_variables - arand_without_variables


    mean = np.mean(all_delta_scores, axis=1)
    std  =  np.std(all_delta_scores, axis=1)
    plt.figure()
    plt.errorbar(sigmas_array, mean[:,0], yerr=std[:,0], label=r'$\alpha$ and $\beta$ have random directions')
    plt.errorbar(sigmas_array, mean[:,1], yerr=std[:,1], label='$\\alpha$ and $\\beta$ are oriented towards\nthe mean of other clusters')
    plt.grid("True")
    plt.title("How much the rand score changes when the model\n considers the (continuous) exogenous variables")
    plt.xscale('log')
    plt.xlabel(r"$\Sigma$")
    plt.ylabel(r"$\Delta$ adjusted rand score")
    plt.legend()















