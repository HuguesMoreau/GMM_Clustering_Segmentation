import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans  # the initialization might require kmeans

from utils.logistic_regression import logistic_regression, softmax
from utils.nan_operators import nan_cov, nan_logsumexp

from utils.data_generation import generate_data  # for testing
from utils.exogenous_utils import explain_contributions_exogenous
from visualizations.clustering_results import generate_colors, plot_cluster_ind_day
# from visualizations.cluster_explanation import compute_plot_explanations



#%%

class GMM_segmentation():
    def __init__(self, dim:int, n_clusters:int, n_segments:int,
                 n_variables:tuple=(0, 0, 0),
                 covariance_type:str="spherical", min_slope=None):
        """
        Parameters
        ----------
        dim: int
        n_clusters: int
        n_segments: int
        n_variables: triple of ints, for individual, temporal, and mixed (default value is (0, 0, 0))
        covariance_type: str, optional. Must be one of:
            'full': each component has its own general covariance matrix.
            'tied': all components share the same general covariance matrix.
            'diag': each component has its own diagonal covariance matrix.
            'spherical' (default): each component has its own single variance.
            (behaves similarly to the sklearn implementation)
        min_slope: float or None, optional
            The minimal change of segments over time.
            Default is None, which corresponds to no restrictions.
        """
        n_individual_variables, n_temporal_variables, n_mixed_variables = n_variables

        self.dim = dim
        self.n_clusters = n_clusters
        self.n_segments = n_segments
        self.n_ind_variables = n_individual_variables
        self.n_temp_variables = n_temporal_variables
        self.n_mixed_variables = n_mixed_variables
        self.min_slope = min_slope


        self.pi = np.zeros(n_clusters) + 1/n_clusters
        self.u = np.zeros((n_clusters, n_segments))
        self.v = np.zeros((n_clusters, n_segments))
        self.m  = np.random.randn(dim, n_clusters, n_segments)
        self.alpha = np.zeros((n_individual_variables, dim, n_clusters, n_segments))
        self.beta  = np.zeros((n_temporal_variables,   dim, n_clusters, n_segments))
        self.gamma = np.zeros((n_mixed_variables,      dim, n_clusters, n_segments))
        self.covariance_type = covariance_type
        if covariance_type == "spherical":
            self.sigma = np.ones((n_clusters, n_segments))
            self.n_sigma_parameters = n_clusters * n_segments
        elif covariance_type == "tied":
            self.sigma = np.eye(dim)  # shape: (dim, dim)
            self.n_sigma_parameters = int(dim * (dim+1) / 2)
        elif covariance_type == "diag":
            self.sigma = np.ones((dim, n_clusters, n_segments))
            self.n_sigma_parameters = dim * n_clusters * n_segments
        elif covariance_type == "full":
            self.sigma = np.zeros((dim,dim, n_clusters,n_segments)) + np.eye(dim)[:,:,np.newaxis,np.newaxis]
            self.n_sigma_parameters = int(dim * (dim+1) / 2)* n_clusters * n_segments
        else:
            error_msg = f"Unknown covariance type '{covariance_type}'. Must be one of 'full', 'tied', 'diag', 'spherical'."
            raise ValueError(error_msg)

        if min_slope != None:
            self.u += min_slope * np.linspace(-n_segments/2, n_segments/2, n_segments).reshape(1,-1)
            # segment_centers = np.linspace(1/(2*n_segments), 1-1/(2*n_segments), n_segments)
            # segment_borders = np.linspace(0, 1, n_segments+1)
            # segment_centers =(segment_borders[1:] + segment_borders[:-1])/2
            segment_borders = np.linspace(0, 1, n_segments+1)[1:-1]  # shape: (n_clusters, n_segments-1)
            diff_u = np.diff(self.u, axis=1) # shape: (n_clusters, n_segments-1)
            self.v[:,1:] -= np.cumsum(segment_borders*diff_u, axis=1)

            self.u -= self.u.mean()
            self.v -= self.v.mean()




        self.n_parameters = self.m.size + self.alpha.size + self.beta.size + self.gamma.size +\
                            self.n_sigma_parameters + (self.pi.size - 1) +\
                            (self.u.size - self.n_clusters) + (self.v.size - self.n_clusters)
                            # u and v are overparametrized as is
        self.LL_list = []
        self.LL_list_alt = []
        self.proportions_history = []
        self.linear_regression_variance_history = []


    def unpack_vars(self, variables, n_individuals=None, n_days=None):
        """For simplicity, we allow some of the variables to be None. This function
        turns None into useable arrays.
        Note that this function is only about creating arrays with 0 columns to match function signature,
        it does NOT replace NaN values in the arrays"""
        if n_individuals is None: n_individuals = self.n_individuals # number of **training** samples
        if n_days        is None: n_days        = self.n_days        # number of **training** days*

        if variables is None:
            assert self.n_ind_variables + self.n_temp_variables + self.n_mixed_variables == 0, "model expected exogenous variables"
            ind_variables   = np.zeros((n_individuals, 1,      self.n_ind_variables))
            temp_variables  = np.zeros((1,             n_days, self.n_temp_variables))
            mixed_variables = np.zeros((n_individuals, n_days, self.n_mixed_variables))
        else:
            correct_type = (type(variables) == tuple) or (type(variables) == list)
            assert (correct_type & (len(variables) == 3)), "expected three types of variables: individual, temporal, and mixed"
            ind_variables, temp_variables, mixed_variables = variables
            if ind_variables   is None: ind_variables   = np.zeros((n_individuals, 1,      self.n_ind_variables))
            if temp_variables  is None: temp_variables  = np.zeros((1,             n_days, self.n_temp_variables))
            if mixed_variables is None: mixed_variables = np.zeros((n_individuals, n_days, self.n_mixed_variables))
        return (ind_variables, temp_variables, mixed_variables)



    def BIC(self, Y, variables):
        _, log_likelihood = self.E_step(Y, variables)
        n_individuals, n_days,_ = Y.shape
        n_samples = n_individuals  # number of *independant* samples
        n_parameters = self.n_parameters
        return log_likelihood - (n_parameters/2) * np.log(n_samples)


    def ICL(self, Y, variables):
        log_r, _ = self.E_step(Y, variables)
        r_log_r = np.exp(log_r) * log_r
        entropy_term = - np.nansum(r_log_r)
        return self.BIC(Y, variables) - entropy_term


    def N(self, Y, mu, sigma):
        """
        This model's Gaussian distribution has a particular signature

        Parameters
        ----------
        Y: np array with shape  (n_individuals, n_days, dim)
        mu: np array with shape (n_individuals, n_days, dim)
            Contrary to self.m, mu includes the contributions of the exogenous variables
        sigma: shape depends on self.covariance_type
            'full': np array with shape (dim, dim)
            'tied': np array with shape (dim, dim)
            'diag': np array with shape (dim,)
            'spherical': scalar

        Returns
        -------
        log_r: np array with shape (n_individuals, n_days,)
        """
        dim = Y.shape[2]
        epsilon = 1e-15

        if self.covariance_type == "spherical":
            inv_covar = 1/(sigma+epsilon) # this should be a matrix, but broadcasting a scalar is easier
            log_det_sigma = dim * np.log(sigma)   # det_sigma = sigma ** dim
            Y_bar = (Y-mu)
            log_r = (-1/2)* ( (log_det_sigma  + dim * np.log(2*np.pi) )  +\
                              np.sum((Y_bar * inv_covar) * Y_bar, axis=2) )
                 # for 2D matrices, np.diag(A @ B.T) is equal to (A*B).sum(axis=-1)
                 # we use a similar reasoning to replace
                 # np.diag(Y_bar @ inv_covar @ Y_bar.transpose()))

        elif self.covariance_type in ["tied", "full"]:
            inv_covar = np.linalg.pinv(sigma, hermitian=True)   # shape: (dim, dim)
            log_det_sigma = np.log(np.linalg.det(sigma))
            Y_bar = (Y-mu)
            log_r = (-1/2)* ( (log_det_sigma  + dim * np.log(2*np.pi) )  +\
                              np.sum((Y_bar @ inv_covar) * Y_bar, axis=2))

        elif self.covariance_type == "diag":
            inv_covar = (1/(sigma+epsilon))[np.newaxis,np.newaxis,:]  # shape: (1,1,dim)
            log_det_sigma = np.sum(np.log(sigma))
            Y_bar = (Y-mu)
            log_r = (-1/2)* ( (log_det_sigma  + dim * np.log(2*np.pi) )  +\
                              np.sum((Y_bar * inv_covar) * Y_bar, axis=2))

        return log_r



    def LogLikelihood(self, Y, variables):
        """
        This function is made redundant by the computation of the log likelihood
        in the E-step method, and hence is unused, but remains useful for debugging.
        However, it is noticeably unoptimized.
        Parameters
        ----------
        Y: np array with shape (n_individuals, n_days, dim)
        variables: optional, triple of:
            ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
            temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
            mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None
        Returns
        -------
        log_likelihood: scalar
            The computation of the log_likelihood is greatly simplified using
            some intermediary variables we compute here.
        """
        n_individuals, n_days, _ = Y.shape
        ind_variables, temp_variables, mixed_variables = self.unpack_vars(variables, n_individuals, n_days)

        log_kappa = np.zeros((1, n_days, self.n_clusters, self.n_segments))

        log_N = np.zeros((n_individuals, n_days, self.n_clusters, self.n_segments))
        t = np.linspace(0,1, n_days).reshape(-1,1)  # shape: (n_days, 1)

        for k in range(self.n_clusters):
            this_u, this_v = self.u[k,:].reshape(1,-1), self.v[k,:].reshape(1,-1) # shape: (1, n_segments)
            log_kappa[0,:,k,:] = this_u * t + this_v                              # log_kappa.shape: (1, n_days, n_clusters, n_segments)
            log_kappa[0,:,k,:] -= logsumexp(log_kappa[0,:,k,:], axis=1, keepdims=True)

            for s in range(self.n_segments):
                mu_ks = self.m[:,k,s][np.newaxis, np.newaxis,:]  +\
                    (ind_variables   @ self.alpha[:,:,k,s]) +\
                    (temp_variables  @  self.beta[:,:,k,s]) +\
                    (mixed_variables @ self.gamma[:,:,k,s])  # shape: (n_individuals, n_days, dim)
                sigma_ks = self.sigma if self.covariance_type == 'tied' else  self.sigma[...,k,s]
                log_N[:,:,k,s] = self.N(Y, mu_ks, sigma_ks)  # shape: (n_individuals, n_days)


        logsum_kappa_N = nan_logsumexp(log_kappa + log_N, axis=3, all_nan_return=np.nan)
        log_rho = np.log(self.pi[np.newaxis,k]) + np.nansum(logsum_kappa_N, axis=1)  # shape: (n_individuals, n_clusters)
        log_likelihood = np.nansum(nan_logsumexp(log_rho, axis=1, all_nan_return=np.nan), axis=0)

        return log_likelihood



    def get_residuals(self, Y, variables):
        """
        Parameters
        ----------
        Y: np array with shape (n_individuals, n_days, dim)
        variables: optional, triple of:
            ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
            temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
            mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None

        Returns
        -------
        residuals : np array with shape (n_individuals, n_days, dim, n_clusters, n_segments)
            How far the samples are from each segment mean, positive or negative (Y - mean)
            Note that the residuals do not care about the standard deviation.
        /!\ This array is likely to be big. In practice, we only use this function in dimension 1,
        with one cluster and one segment (pure linear regression).
        """
        n_individuals, n_days, dim = Y.shape
        ind_variables, temp_variables, mixed_variables = self.unpack_vars(variables, n_individuals, n_days)
        residuals = np.zeros((n_individuals, n_days, dim, self.n_clusters, self.n_segments))

        for k in range(self.n_clusters):
            for s in range(self.n_segments):
                mu_ks = self.m[:,k,s][np.newaxis, np.newaxis,:]  +\
                    (ind_variables   @ self.alpha[:,:,k,s]) +\
                    (temp_variables  @  self.beta[:,:,k,s]) +\
                    (mixed_variables @ self.gamma[:,:,k,s])  # shape: (n_individuals, n_days, dim)
                residuals[:,:,:,k,s] = Y - mu_ks
        return residuals






    def E_step(self, Y, variables):
        """
        Parameters
        ----------
        Y: np array with shape (n_individuals, n_days, dim)
        variables: optional, triple of:
            ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
            temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
            mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None


        Returns
        -------
        log_r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
            the natural log of the probability of belonging to a cluster for a
            couple (individual, day)
        log_likelihood: scalar
            The computation of the log_likelihood is greatly simplified using
            some intermediary variables we compute here.

        """
        n_individuals, n_days, _ = Y.shape
        ind_variables, temp_variables, mixed_variables = self.unpack_vars(variables, n_individuals, n_days)

        log_kappa = np.zeros((1, n_days, self.n_clusters, self.n_segments))
        log_N = np.zeros((n_individuals, n_days, self.n_clusters, self.n_segments))
        t = np.linspace(0,1, n_days).reshape(-1,1)  # shape: (n_days, 1)

        for k in range(self.n_clusters):
            this_u, this_v = self.u[k,:].reshape(1,-1), self.v[k,:].reshape(1,-1) # shape: (1, n_segments)
            log_kappa[0,:,k,:] = this_u * t + this_v                              # log_kappa.shape: (1, n_days, n_clusters, n_segments)
            log_kappa[0,:,k,:] -= logsumexp(log_kappa[0,:,k,:], axis=1, keepdims=True)

            for s in range(self.n_segments):
                mu_ks = self.m[:,k,s][np.newaxis, np.newaxis,:]  +\
                    (ind_variables   @ self.alpha[:,:,k,s])  +\
                    (temp_variables  @  self.beta[:,:,k,s])  +\
                    (mixed_variables @ self.gamma[:,:,k,s])  # shape: (n_individuals, n_days, dim)
                sigma_ks = self.sigma if self.covariance_type == 'tied' else  self.sigma[...,k,s]
                log_N[:,:,k,s] = self.N(Y, mu_ks, sigma_ks)  # shape: (n_individuals, n_days)


        logsum_kappa_N = nan_logsumexp(log_kappa + log_N, axis=3, all_nan_return=np.nan)
        log_rho = np.log(self.pi[np.newaxis,:]) + np.nansum(logsum_kappa_N, axis=1)  # shape: (n_individuals, n_clusters)
        log_rho -= nan_logsumexp(log_rho, axis=1, keepdims=True, all_nan_return=np.nan) # shape: (n_individuals, n_clusters)
        log_rho  = log_rho[:,np.newaxis,:,np.newaxis]                                   # shape: (n_individuals, 1, n_clusters, 1)

        log_posterior_segment_knowing_cluster = log_kappa + log_N      # shape: (n_individuals, n_days, n_clusters, n_segments)
        log_posterior_segment_knowing_cluster -= nan_logsumexp(log_posterior_segment_knowing_cluster, axis=3, keepdims=True)

        log_r = log_rho + log_posterior_segment_knowing_cluster

        LL_per_ind_cluster = np.log(self.pi[np.newaxis,:]) + np.nansum(nan_logsumexp(log_kappa + log_N, axis=3, all_nan_return=np.nan), axis=1)   # shape: (n_individuals, n_clusters)
        log_likelihood = np.nansum(nan_logsumexp(LL_per_ind_cluster, axis=1, all_nan_return=np.nan), axis=0)

        return log_r, log_likelihood




    def M_step(self, Y, variables, log_r):
        """
        Parameters
        ----------
        Y: np array with shape (n_individuals, n_days, dim)
        variables: optional, triple of:
            ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
            temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
            mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None
        log_r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
            the natural log of the probability of belonging to a cluster for a
            couple (individual, day)
        Returns
        -------
        None (replaces the values in the parameters with their values at the next iteration)
        """

        n_individuals, n_days, _ = Y.shape
        ind_variables, temp_variables, mixed_variables = self.unpack_vars(variables, n_individuals, n_days)

        count = np.nansum(np.exp(log_r), axis=(0,1, 3)) + np.finfo(float).eps   # shape: (n_clusters)
        count += 1/2 # Dirichlet prior with alpha = 1/2
        self.pi = count/ np.nansum(count)                                        # shape: (n_clusters)


        new_u = np.zeros_like(self.u)  # shape: (n_clusters, n_segments)s
        new_v = np.zeros_like(self.v)  # shape: (n_clusters, n_segments)
        for k in range(self.n_clusters):
            sum_r = np.exp(nan_logsumexp(log_r[:,:,k,:], axis=0, all_nan_return=np.nan)) # shape: (n_days, n_segments)
            sum_r += 1e-3
            sum_r = sum_r/np.nansum(sum_r, axis=1, keepdims=True)
            if len(self.LL_list) > 1:
                new_u[k,:], new_v[k,:] = logistic_regression(sum_r, min_slope=self.min_slope, init=(self.u[k,:], self.v[k,:]))
            else:
                new_u[k,:], new_v[k,:] = logistic_regression(sum_r, min_slope=self.min_slope)
        self.u, self.v = new_u, new_v

        self.m, self.alpha, self.beta, self.gamma = explain_contributions_exogenous(Y, (ind_variables, temp_variables, mixed_variables), log_r)


        # Computation of the covariance
        proportion_variance = np.zeros((self.n_clusters, self.n_segments))
        new_sigma = np.zeros_like(self.sigma)
        for k in range(self.n_clusters):
            for s in range(self.n_segments):
                if self.CEM:
                    this_cluster_r = np.exp(log_r[:,:,k,s] )     # shape: (n_ind, n_days)
                    this_cluster_weight = this_cluster_r / (np.nansum(this_cluster_r) + 1e-15)
                    this_cluster_weight = this_cluster_weight[:,:,np.newaxis]  # shape: (n_ind, n_days, 1), sums to one
                else:

                    normalized_log_r = log_r[:,:,k,s] - nan_logsumexp(log_r[:,:,k,s], axis=(0,1))  # shape: (n_ind, n_days)
                    this_cluster_weight = np.exp(normalized_log_r)[:,:,np.newaxis]  # shape: (n_ind, n_days, 1), sums to one

                    this_cluster_weight /= (np.nansum(this_cluster_weight) +1e-10)
                        # when the log likelihood is extremely small, numerical problem may ensue:
                        # the values of this_cluster_weight may be in {0,1}, meaning the sum of its weights
                        # is above 1. Normalizing by the sum a second time seems clumsy, but it is effective.

                assert np.isclose(np.nansum(this_cluster_weight), 1) or (self.CEM and np.allclose(this_cluster_weight, 0))

                Y_bar = Y.copy() - self.m[:,k,s].reshape(1,1,-1)
                var_per_axis = np.nansum(this_cluster_weight * (Y_bar)**2, axis=(0,1)) # shape: (dim,)
                total_variance = np.nansum(var_per_axis)
                Y_bar -= (ind_variables   @ self.alpha[:,:,k,s])
                Y_bar -= (temp_variables  @ self.beta[ :,:,k,s])
                Y_bar -= (mixed_variables @ self.gamma[ :,:,k,s])
                variance_after_regression = np.nansum(this_cluster_weight * (Y_bar)**2)
                proportion_variance[k, s] = 1 - variance_after_regression/(total_variance + 1e-10)


                if self.covariance_type == "spherical":
                    sigma_per_dimension = np.nansum(this_cluster_weight * (Y_bar**2), axis=(0,1))
                        # here, we use sum() instead of mean() because this_cluster_weight already sums to one
                        # it plays the role of the 1/n
                    new_sigma[k,s] = np.nanmean(sigma_per_dimension) + np.finfo(float).eps  # adding an epsilon for stability
                elif self.covariance_type == "diag":
                    sigma_per_dimension = np.nansum(this_cluster_weight * (Y_bar**2), axis=(0,1))
                    new_sigma[:,k,s] = sigma_per_dimension + np.finfo(float).eps

                else:
                    if np.isnan(Y.any()): raise NotImplementedError("'full' and 'tied' covariances are not available with missing values")
                    this_cov = np.tensordot(this_cluster_weight * Y_bar, Y_bar, axes=[[0,1],[0,1]])
                    this_cov += np.finfo(float).eps  * np.eye(self.dim) # regularization
                    if self.covariance_type == "full":
                        new_sigma[:,:,k,s] = this_cov

                    elif self.covariance_type == "tied":
                        new_sigma += self.pi[k] * this_cov

                del Y_bar


        self.sigma = new_sigma
        self.linear_regression_variance_history.append(proportion_variance)






    def EM_init(self, Y, variables, init_method, init_with_CEM=False):
        """
        Parameters
        ----------
        Y: np array with shape (n_individuals, n_days, dim)
        variables: optional, triple of:
            ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
            temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
            mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None
          The exogenous variables need at least to have the same variance, which will
          happen in practice because we will standard-scale them.
        init_method: couple of (string, object). The first string is either 'parameters' or 'responsibilities'
            depending on the initialization method. The type the second element of the couple depends on
            the type of init:
            - "parameters": object is a couple of strings  The first string applies to the cluster, the
                other, to the segments. Possible strings are, for clusters|segments
                - "kmeans" (where individuals|days play the role of samples, and (hours*days)|shours are features)
                - "random" draw from a Gaussan where the mean and covariance are the ones of the provided dataset Y
                - "uniform" (for segments only): divides the time axis into n_segments uniform segments, and compute the
                    mean per segment.
                Note that the intitialization for the exogenous variables' contributions and covariance
                is not changed by any of the parameters: they are zero and identity, respectively.
            - "responsibilities": the argument is a np array of log-probabilities, with shape (n_ind, n_days, n_clusters, n_segments)

        init_with_CEM: bool, optional (defaults to False)
            if True, begin by applying Classification EM to initialize the parameters, before using regular EM.
            In this case, the init_method parameter applies to the initialization of the CEM loop.

        Returns
        -------
        None.
        """

        if init_with_CEM:
            old_CEM = self.CEM   # self.CEM will be set to True
            self.EM(Y, variables, init_method, CEM=True, init_with_CEM=False,
                    print_every=0)
            self.CEM = old_CEM
            # reset some of the variables
            self.LL_list_CEM = self.LL_list
            self.LL_list = []
            self.LL_list_alt = []
            self.proportions_history = []
            self.linear_regression_variance_history = []
            return


        init_type, init_arg = init_method
        n_individuals, n_days, _ = Y.shape
        ind_variables, temp_variables, mixed_variables = self.unpack_vars(variables, n_individuals, n_days)

        if init_type == "parameters":
            cluster_init_method, segment_init_method = init_arg
            n_individuals, n_days, dim = Y.shape

                # Clusters
            if cluster_init_method == "kmeans":
                kmeans = KMeans(n_clusters = self.n_clusters)
                mean_per_ind = np.nanmean(Y, axis=1)           # shape: (n_individuals, dim)
                nan_inds = np.isnan(mean_per_ind).any(axis=1)     # shape: (n_individuals,)
                cluster_labels = np.zeros(n_individuals).astype(int) - 1   # default cluster = -1 (this value
                    # will be replaced for all non-NaN individuals)
                cluster_labels[~nan_inds] = kmeans.fit_predict(mean_per_ind[~nan_inds,:])
                cluster_centers = kmeans.cluster_centers_.T   # shape: (dim, n_clusters)

            elif cluster_init_method == "random":
                mean = np.nanmean(Y, axis=(0,1)) # shape: (dim,)
                covariance = nan_cov(Y.reshape(-1, Y.shape[-1]))
                sqrt_cov = sqrtm(covariance) if covariance.size > 1 else np.array(np.sqrt(covariance)).reshape(1,1)

                # the next step will require all clusters to be non-empty, but only if n_segments > 1
                do_while_condition = True
                n_iter, max_iter = 0, 100
                while do_while_condition and n_iter < max_iter:
                    n_iter += 1
                    cluster_centers = sqrt_cov @ np.random.randn(self.dim, self.n_clusters) + mean[:,np.newaxis]  # shape: (dim, n_clusters)
                    if self.n_segments == 1: do_while_condition = False  # we only need non-empty clusters if  n_segments > 1
                    distances = np.linalg.norm(Y.reshape(-1, Y.shape[2])[:,:,np.newaxis] - cluster_centers[np.newaxis,:,:], axis=1)  # shape: (n_individuals*n_days, n_clusters)
                    distances[np.isnan(distances)] = np.inf
                    cluster_labels = np.argmin(distances, axis=1) # shape: (n_individuals * n_days, n_clusters)
                    cluster_labels = stats.mode(cluster_labels.reshape(n_individuals, n_days), axis=1, keepdims=False)[0] # mode along time
                    # ... while set(np.unique(cluster_labels)) != set(range(self.n_clusters))
                    if set(np.unique(cluster_labels)) == set(range(self.n_clusters)):
                        do_while_condition = False

                if n_iter == max_iter:  # previous init failed
                    print("\t\t\t\t init failed")
                    empty_clusters = np.where(cluster_labels == 0)
                    for k in empty_clusters:
                        cluster_centers[k,:] = (sqrt_cov /10) @ np.random.randn(self.dim) + mean[:,np.newaxis]
                    distances = np.linalg.norm(Y.reshape(-1, Y.shape[2])[:,:,np.newaxis] - cluster_centers[np.newaxis,:,:], axis=1)  # shape: (n_individuals*n_days, n_clusters)
                    distances[np.isnan(distances)] = np.inf
                    cluster_labels = np.argmin(distances, axis=1) # shape: (n_individuals * n_days, n_clusters)
                    cluster_labels = stats.mode(cluster_labels.reshape(n_individuals, n_days), axis=1, keepdims=False)[0] # mode along time


            else:
                raise ValueError(f"Unknown cluster initiamization method ('{cluster_init_method}'). Available methods are 'kmeans' and 'random'.")

                # Segments
            if self.n_segments == 1:
                self.m[:,:,0] = cluster_centers
            else: # self.n_segments > 1:
                if segment_init_method == "kmeans":
                    for k in range(self.n_clusters):
                        Y_this_k = Y[cluster_labels == k,:,:]             # shape: (n_individuals, n_days, dim)
                        mean_per_day = np.nanmean(Y_this_k, axis=0)       # shape: (n_days, dim)
                        nan_days = np.isnan(mean_per_day).any(axis=1)     # shape: (n_days,)
                        kmeans = KMeans(n_clusters = self.n_segments)
                        kmeans.fit(mean_per_day[~nan_days,:])
                        self.m[:,k,:] = kmeans.cluster_centers_.T   # shape: (dim, n_segments)

                    global_std = np.nanmean(np.nanstd(Y, axis=(0,1)))

                elif segment_init_method == "random":
                    for k in range(self.n_clusters):
                        Y_this_k = Y[cluster_labels == k,:,:]
                        mean = np.nanmean(Y_this_k, axis=(0,1)) # shape: (dim,)
                        covariance = nan_cov(Y_this_k.reshape(-1, Y_this_k.shape[-1]))
                        sqrt_cov = sqrtm(covariance) if covariance.size > 1 else np.array(np.sqrt(covariance)).reshape(1,1)
                        segment_labels = np.zeros(n_days) -1
                        while set(np.unique(segment_labels)) != set(range(self.n_segments)): # retry as many times as needed
                            segment_centers = sqrt_cov @ np.random.randn(self.dim, self.n_segments) + mean[:,np.newaxis]  # shape: (dim, n_segments)
                            distances = np.linalg.norm(np.nanmean(Y_this_k, axis=0)[:,:,np.newaxis] - segment_centers[np.newaxis,:,:], axis=1)  # shape: (n_days, n_segments)
                            segment_labels = np.argmin(distances, axis=1)
                        self.m[:,k,:] = segment_centers

                elif segment_init_method == "uniform":
                    for k in range(self.n_clusters):
                        Y_this_k = Y[cluster_labels == k,:,:]
                        day_array = np.arange(n_days)
                        segment_borders = np.linspace(0, n_days, self.n_segments+1)
                        for s in range(self.n_segments):
                            selected_days = (day_array >= segment_borders[s]) & (day_array < segment_borders[s+1])
                            Y_this_segment = Y[:,selected_days,:]
                            self.m[:,k,s] = np.nanmean(Y_this_segment, axis=(0,1)) # shape: (dim,)

                    global_std = np.nanmean(np.nanstd(Y, axis=(0,1)))
                    self.m += np.random.randn(*self.m.shape) * global_std * 0.2
                else:
                    raise ValueError(f"Unknown segment initialization method ('{segment_init_method}'). Available methods are 'kmeans', 'random', and 'uniform'.")

        elif init_type == "responsibilities":
            log_r = init_arg
            self.M_step(Y, variables, log_r)

        else:
            raise ValueError(f"Uninkown init type '{init_type}'. Choose either 'responsibilities' or 'parameters'")



    def EM(self, Y, variables, init_method, CEM=False, init_with_CEM=False,
           patience=10, print_every=10):
        """
        Parameters
        ----------
        Y: np array with shape (n_individuals, n_days, dim)
        variables: optional, triple of:
            ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
            temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
            mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None
          The exogenous variables need at least to have the same variance, which will
          happen in practice because we will standard-scale them.
        init_method: couple of strings. The first string applies to the cluster, the other, to the segments.
            Possible strings are, for clusters|segments
            - "kmeans" (where individuals|days play the role of samples, and (hours*days)|shours are features)
            - "random" draw from a Gaussan where the mean and covariance are the ones of the provided dataset Y
            - "uniform" (for segments only): divides the time axis into n_segments uniform segments, and compute the
                mean per segment.
            Note that the intitialization for the exogenous variables' contributions and covariance
            is not changed by any of the parameters: they are zero and identity, respectively.
        CEM: bool, optional
            If True, use Classification EM instead of EM, which consists in assigning
            one to the highest responsibilities and zero to the others.
            Defaults to False
        init_with_CEM: bool, optional (defaults to False)
            if True, begin by applying Classification EM to initialize the parameters, before using regular EM.
            In this case, the init_method parameter applies to the initialization of the CEM loop.
        patience: int, optional
            We stop iterating if the log-likelihood did not increase by more than a threshold
            in less that [patience] iterations. Defaults to 10.
            This value can be set to zero or a negative number to make sire the algorithm iterates only once.
        print_every: positive int, optional
            EM prints the log likely hood every Y iterations.
            If it is set to zero, the function prints nothing.
            Defaults to 10.

        Returns
        -------
        LL_list: list of (negative) floats
        """

        self.n_individuals, self.n_days, _ = Y.shape
            # will be used to compute a BIC

            # initialization:
        self.CEM = CEM
        self.EM_init(Y, variables, init_method, init_with_CEM)

            # EM-loop
        self.LL_list = []
        best_LL  = -np.inf
        i = 0
        iter_without_increasing = 0


        while iter_without_increasing < patience and i < 1e4 :
            log_r, current_LL = self.E_step(Y, variables)
            if self.CEM:
                r_classified = (log_r == np.max(log_r, axis=(2,3), keepdims=True))
                log_r_classified = np.zeros_like(log_r)
                log_r_classified[r_classified == 0] = -np.inf
                    # the -inf values will cancel out when computing the exponential
                log_r = log_r_classified
                del log_r_classified, r_classified

            if print_every > 0:
                if i%print_every == 0: print(f'\titeration {i}, LL={current_LL:.3e}')

            self.LL_list.append(current_LL)

            self.M_step(Y, variables, log_r)
            self.proportions_history.append(np.copy(self.pi))

            improvement = current_LL - best_LL > np.abs(best_LL * 1e-4) if i > 0 else True
            if improvement:
                best_LL = current_LL
                iter_without_increasing = 0
            else:
                iter_without_increasing += 1
            i += 1
        return self.LL_list
















#%%

if __name__ == "__main__":


    seed =  np.random.randint(1,10001)
    # seed = 249
    # seed = 5923
    np.random.seed(seed)
    print("="*40+f"\n\t\tSEED = {seed}\n"+"="*40)
    n_clusters = 4
    n_segments = 4
    n_ind_variables   = 2
    n_temp_variables  = 2
    n_mixed_variables = 0
    n_individuals = 300
    n_days = 300
    dim = 12

    base_colors = generate_colors(n_clusters*n_segments, method="random")

    pi_gt = np.ones(n_clusters) / n_clusters

    u_gt, v_gt = np.zeros((n_clusters, n_segments)), np.zeros((n_clusters, n_segments))

    u_gt[:,:] = np.linspace(-500, 500, n_segments+2)[np.newaxis,1:-1]
    delta_u = (u_gt[:,-2:-1] - u_gt[:,0:1])/u_gt.shape[1] if n_segments > 1 else 1.
    u_gt += np.random.uniform(-1, 1, u_gt.shape) * delta_u
    v_gt[:,0] = 0 # left at zero
    for k in range(n_clusters):
        segment_borders = np.linspace(0,1, n_segments+1)[1:-1]  # shape: (n_segments-1, )
        segment_borders += 2*np.random.uniform(-1/(3*n_segments), 1/(3*n_segments), size=segment_borders.shape)
        for s in range(1, n_segments):
            v_gt[k,s] = v_gt[k,s-1] + segment_borders[s-1] * (u_gt[k,s-1] - u_gt[k,s])
    m_gt =     np.random.randn(                  dim, n_clusters, n_segments) * 2
    alpha_gt = np.random.randn(n_ind_variables,   dim, n_clusters, n_segments) * 1
    beta_gt  = np.random.randn(n_temp_variables,  dim, n_clusters, n_segments) * 1
    gamma_gt = np.random.randn(n_mixed_variables, dim, n_clusters, n_segments) * 1
    sigma_gt = np.ones((n_clusters, n_segments)) * 1.

    ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
    temp_variables  = np.random.randn(1,             n_days, n_temp_variables)
    mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)
    variables = (ind_variables, temp_variables, mixed_variables)
    contributions_gt = alpha_gt, beta_gt, gamma_gt
    Y, r_gt = generate_data(pi_gt, u_gt, v_gt, m_gt, contributions_gt, sigma_gt, variables)
    # reorder the individuals so that clusters are aligned

    cluster_assigment_one_hot = r_gt.sum(axis=(1,3))                 # shape: (n_individuals, n_clusters)
    cluster_assigment = np.argmax(cluster_assigment_one_hot, axis=1) # shape: (n_individuals,)
    individual_reordering = np.argsort(cluster_assigment)
    Y    = Y[   individual_reordering,:,:]
    r_gt = r_gt[individual_reordering,:,:,:]


    Y[0, 1, :] = np.nan
    Y[:, 0, :] = np.nan

    plt.figure(figsize=(15, 15))
    plt.subplot(1,2,1)
    plot_cluster_ind_day(r_gt, colors_per_cluster=base_colors, title='ground truth', ind_ordering='original', new_figure=False)
    t = np.linspace(0,1, n_days).reshape(-1,1)

    for k in range(n_clusters):
        plt.subplot(n_clusters, 2, 2*k+2)
        this_u, this_v = u_gt[k,:].reshape(1,-1), v_gt[k,:].reshape(1,-1) # shape: (1, n_segments)
        probability = softmax(this_u * t + this_v, axis=1)         # shape: (n_days, n_segments), sums to one along segments
        for s in range(n_segments):
            plt.plot(probability[:,s], c=base_colors[k*n_segments+s,:])

    # Model testing : using the variables
    CEM = {True, False}
    init_clusters = {'kmeans', 'random'}
    init_segments = {'kmeans', 'random', 'uniform'}


    model = GMM_segmentation(dim, n_clusters=n_clusters, n_segments=n_segments,
                                  n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                              min_slope=100)
    LL_list = model.EM(Y, variables, init_method=('parameters', ("random","kmeans")), init_with_CEM=False,
                        print_every=1, CEM=False)
    estimated_r = np.exp(model.E_step(Y, variables)[0])


    plt.figure(figsize=(15, 15))
    plt.subplot(1,2,1)
    plot_cluster_ind_day(estimated_r, colors_per_cluster=base_colors[:model.n_clusters*model.n_segments,:],
                         ind_ordering='original', new_figure=False, title='model assignment')
    t = np.linspace(0,1, n_days).reshape(-1,1)

    for k in range(model.n_clusters):
        plt.subplot(model.n_clusters, 2, 2*k+2)
        this_u, this_v = model.u[k,:].reshape(1,-1), model.v[k,:].reshape(1,-1) # shape: (1, n_segments)
        probability = softmax(this_u * t + this_v, axis=1) # shape: (n_days, n_segments), sums to one along segments
        for s in range(model.n_segments):
            plt.plot(probability[:,s], c=base_colors[k*n_segments+s,:])



    plt.figure(figsize=(13, 13))
    plt.plot(LL_list)
    plt.title("Log_likelihood of the model $using$ the exogenous variables")
    plt.xticks(np.arange(len(LL_list)), np.arange(len(LL_list)))
    plt.grid(True)









