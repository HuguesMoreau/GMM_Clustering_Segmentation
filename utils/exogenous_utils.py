"""
Contains two functions to deal with exogenous variables
(estimate their contribution for the M step and explain the clusters)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.special import logsumexp
from utils.nan_operators import nan_logsumexp


#%%


def sklearn_explain_contributions_exogenous(X, variables, log_r):
    """
    This function has the same signature and behaviour as explain_contributions_exogenous,
    except it computes an explicit cross-product of variables, which strains the memory
    We keep it for testing purposes
    """
    ind_variables, temp_variables, mixed_variables = variables
    n_ind_variables   =   ind_variables.shape[2]
    n_temp_variables  =  temp_variables.shape[2]
    n_mixed_variables = mixed_variables.shape[2]
    n_individuals, n_days, dim = X.shape
    _, _, n_clusters, n_segments = log_r.shape

    alpha = np.zeros((n_ind_variables,   dim, n_clusters, n_segments))
    beta  = np.zeros((n_temp_variables,  dim, n_clusters, n_segments))
    gamma = np.zeros((n_mixed_variables, dim, n_clusters, n_segments))
    m     = np.zeros((                  dim, n_clusters, n_segments))

    X_reshaped = X.reshape(-1, X.shape[2])
    n_exogenous_variables = n_ind_variables + n_temp_variables + n_mixed_variables
    all_variables = np.zeros((n_individuals * n_days, n_exogenous_variables))
    all_variables[:, :n_ind_variables]  = np.repeat(ind_variables[:,0,:], repeats=n_days, axis=0)
    all_variables[:,n_ind_variables:n_exogenous_variables-n_mixed_variables] = np.tile( temp_variables[0,:,:], reps = (n_individuals, 1))
    all_variables[:,n_exogenous_variables-n_mixed_variables: ] = mixed_variables.reshape(n_individuals * n_days, -1)

    for k in range(n_clusters):
        for s in range(n_segments):
            lr = LinearRegression(n_jobs=-2)
            log_sample_weight = log_r[:,:,k,s].reshape(-1).copy()
            log_sample_weight -= nan_logsumexp(log_sample_weight, axis=0, keepdims=True)
            sample_weight = np.exp(log_sample_weight)
            sample_weight[np.isnan(sample_weight)] = 0
            if np.isnan(X_reshaped).any(): # we need to replace the values by means, but not in-place (because the
                X_sklearn = X_reshaped.copy()            # weights to compute the mean change with clusters)
                for d in range(dim):
                    this_dim = X_sklearn[:,d]
                    this_dim[np.isnan(this_dim)] = np.nansum(this_dim * sample_weight)  # sum with weights that sum to 1 = mean
                    X_sklearn[:,d] = this_dim
            else:
                X_sklearn = X_reshaped # do nothing
            lr.fit(all_variables, X_sklearn, sample_weight=sample_weight)  # be careful, X here is the 'output' of the linear model

            alphabetagamma = lr.coef_.T  # shape: (n_ind_variables + n_temp_variables, dim)
            alpha[:,:,k,s] = alphabetagamma[:n_ind_variables,:]
            beta[ :,:,k,s] = alphabetagamma[n_ind_variables:n_exogenous_variables-n_mixed_variables,:]
            gamma[:,:,k,s] = alphabetagamma[n_exogenous_variables-n_mixed_variables:,:]
            m[:,k,s] = lr.intercept_

    return m, alpha, beta, gamma





#%%

def half_computed_covar(variables_bar, weights):
    """
    this is equivalent to:
    np.nansum(variables_bar[:,:,:,np.newaxis] * weights * variables_bar[:,:,np.newaxis,:], axis=(0,1))
    except that we do not compute the extradiagonal terms twice (the matrix is symmetric).
    This function exists only to save time.

    Parameters
    ----------
    variables_bar: np array with shape (n_individuals|1, n_days|1, n_variables)
        Can correspond to individual, temporal, or mixed variables.
        May contain NaNs, but must be centered. (the function does not check this)
        /!\ Contrary to other function, variables_bar is not a triple
    weights: np array with shape (n_individuals, n_days, 1, 1) that sum to 1.

    Returns
    -------
    cov: np array with shape (n_variables, n_variables), symmetric
        (equivalent to computing with ddof = n_individuals*n_days)
    """
    n_individuals, n_days, _, _ = weights.shape
    n_variables = variables_bar.shape[2]
    cov = np.zeros((n_variables, n_variables))

    for d1 in range(n_variables):
        cov[d1,:d1+1] = np.nansum(variables_bar[:,:,d1,np.newaxis] *\
                                  weights[:,:,:,0] *\
                                  variables_bar[:,:,:d1+1], axis=(0,1))

    for d1 in range(n_variables):
        for d2 in range(d1):
            cov[d2, d1] = cov[d1, d2]


    return cov






if __name__ == "__main__":

    n_ind_variables   = 2
    n_temp_variables  = 3
    n_mixed_variables = 4
    n_individuals = 5   # we dont want too many individuals/days to avoid covariances to be too close to 1.
    n_days = 6

    ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
    temp_variables  = np.random.randn(   1,          n_days, n_temp_variables)
    mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)

    weights = np.random.uniform(0,1, size=(n_individuals, n_days, 1, 1))
    weights /= weights.sum()

    ind_variables_bar   = ind_variables   - np.mean(  ind_variables, axis=(0,1), keepdims=True)
    temp_variables_bar  = temp_variables  - np.mean( temp_variables, axis=(0,1), keepdims=True)
    mixed_variables_bar = mixed_variables - np.mean(mixed_variables, axis=(0,1), keepdims=True)


    for variables_bar in [ind_variables_bar, temp_variables_bar, mixed_variables_bar]:
        cov_parcimonious = half_computed_covar(variables_bar, weights)
        cov_complete = np.nansum(variables_bar[:,:,:,np.newaxis] * weights * variables_bar[:,:,np.newaxis,:], axis=(0,1))

        assert np.allclose(cov_complete, cov_parcimonious)



#%%


    #   Time measurement (make sure we actually save time)

    redo_time_measurement = True  # should be less than a minute

    if redo_time_measurement:

        n_ind_variables   = 10
        n_temp_variables  = 10
        n_mixed_variables = 10
        n_individuals = 1000
        n_days = 1000

        import time
        time_parcimonious = 0
        time_complete = 0


        n_trials = 10
        for _ in range(n_trials):

            ind_variables_bar   = np.random.randn(n_individuals, 1,      n_ind_variables)
            temp_variables_bar  = np.random.randn(   1,          n_days, n_temp_variables)
            mixed_variables_bar = np.random.randn(n_individuals, n_days, n_mixed_variables)
            # these variables are not centered because we do not care about the result

            weights = np.random.uniform(0,1, size=(n_individuals, n_days, 1, 1))
            weights /= weights.sum()


            for variables_bar in [ind_variables_bar, temp_variables_bar, mixed_variables_bar]:
                start_time = time.time()
                cov_parcimonious = half_computed_covar(variables_bar, weights)
                time_parcimonious += time.time() - start_time

                start_time = time.time()
                cov_complete = np.nansum(variables_bar[:,:,:,np.newaxis] * weights * variables_bar[:,:,np.newaxis,:], axis=(0,1))
                time_complete += time.time() - start_time


        print(f"With parcimonious function: {time_parcimonious/n_trials:.2f} s")
        print(f"With baseline function: {time_complete/n_trials:.2f} s")
        print(f"  ({((time_complete - time_parcimonious)/ time_complete) * 100:.2f} % improvement) \n")

    # With parcimonious function: 1.11 s
    # With baseline function: 1.50 s
    #   (26.20 % improvement)



#%%





def explain_contributions_exogenous(X, variables, log_r):
    """
    Parameters
    ----------
    X: np array with shape (n_individuals, n_days, dim)
    variables: triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables)
        /!\ Contrary to the regression model, variables are not optional here
    log_r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
        the **log** of the sample weights

    Returns
    -------
    m, alpha, beta, gamma
        m has a shape (dim, n_clusters)
        alpha has a shape (n_ind_variables,   dim, n_clusters)
        beta  has a shape (n_temp_variables,  dim, n_clusters)
        gamma has a shape (n_mixed_variables, dim, n_clusters)
    """
    ind_variables, temp_variables, mixed_variables = variables
    n_ind_variables   =   ind_variables.shape[2]
    n_temp_variables  =  temp_variables.shape[2]
    n_mixed_variables = mixed_variables.shape[2]
    n_individuals, n_days, dim = X.shape
    _, _, n_clusters, n_segments = log_r.shape

    nax = np.newaxis  # shorter notation (we will neex this a lot)

    if n_ind_variables + n_temp_variables + n_mixed_variables == 0:
        # the case where there is only one type of variables (n_ind_variables == 0 XOR n_temp_variables == 0)
        # is already covered by the following code, but the case where no variables is present needs
        # proper care
        m = np.zeros((dim, n_clusters, n_segments))
        alpha = np.zeros((0, dim, n_clusters, n_segments))
        beta  = np.zeros((0, dim, n_clusters, n_segments))
        gamma = np.zeros((0, dim, n_clusters, n_segments))
        for k in range(n_clusters):
            for s in range(n_segments):
                this_cluster_weight = np.exp(log_r[:,:,k,s] - nan_logsumexp(log_r[:,:,k,s], axis=(0,1), keepdims=True))
                m[:,k,s] = np.nansum(this_cluster_weight[:,:,nax] * X, axis=(0,1))
        return m, alpha, beta, gamma
    # end if (end of the particular case)

    dim = X.shape[2]
    n_exo_var = n_ind_variables + n_temp_variables + n_mixed_variables

    alpha = np.zeros((n_ind_variables,   dim, n_clusters, n_segments))
    beta  = np.zeros((n_temp_variables,  dim, n_clusters, n_segments))
    gamma = np.zeros((n_mixed_variables, dim, n_clusters, n_segments))
    m     = np.zeros((                   dim, n_clusters, n_segments))


    for k in range(n_clusters):
        for s in range(n_segments):
            # if (log_weight_ks == -np.inf).any()# CEM may include some values for which r = 0
            log_weight_ks = log_r[:,:,k,s].copy()               # shape: (n_individuals, n_days)
            norm_weights = nan_logsumexp(log_weight_ks, axis=(0,1), keepdims=True)
            if norm_weights == -np.inf: # if all the weights are at zero (meaning all log-weights are -np.inf):
                # (this situation may occur with CER, where the responsibilities are 0/1 )
                assert (log_weight_ks == -np.inf).all()
                break  # alpha, beta, and m stay at zero

            log_weight_ks -= norm_weights

            # Removing empty (zero-weight) individuals or days
            empty_individuals = ~np.any(log_weight_ks > -30, axis=1)  #  shape: (n_individuals,)   exp(-30) = 10**-13
                 # this is almost equivalent to np.all(log_weight_ks < -30), except for NaN values
            nontrivial_X  =             X[~empty_individuals,:,:]
            log_weight_ks = log_weight_ks[~empty_individuals,:]
            empty_days        = ~np.any(log_weight_ks > -30, axis=0)  #  shape: (n_days,)
            nontrivial_X  =  nontrivial_X[:,~empty_days,:]
            log_weight_ks = log_weight_ks[:,~empty_days]

            nontrivial_ind_var   = ind_variables[~empty_individuals,:,:]
            nontrivial_temp_var  = temp_variables[:,~empty_days,:]
            nontrivial_mixed_var = mixed_variables[~empty_individuals,:,:][:,~empty_days,:]


            weight_ks = np.exp(log_weight_ks)[:,:,nax]  # shape: (n_individuals, n_days, 1)

            mean_X = np.nansum(nontrivial_X * weight_ks.astype(np.float32), axis=(0,1), keepdims=True)  # shape: (1, 1, dim)
            X_bar = nontrivial_X - mean_X
            del nontrivial_X  #  X_bar has been weighted and has no individual or days where the weights are all zero

            weight_ind_var  = np.nansum(weight_ks, axis=1, keepdims=True)                            # shape: (n_individuals, 1,      1)
            mean_ind_var  = np.nansum( nontrivial_ind_var * weight_ind_var,  axis=0, keepdims=True)  # shape: (1,             1,      n_ind_variables)
            ind_variables_bar = nontrivial_ind_var - mean_ind_var                                    # shape: (n_individuals, 1,      n_ind_variables)
            del nontrivial_ind_var  #  ind_variables_bar has been weighted and has no useless individuals or days where the weights are all zero
            weight_temp_var = np.nansum(weight_ks, axis=0, keepdims=True)                            # shape: (1,             n_days, 1)
            mean_temp_var = np.nansum(nontrivial_temp_var * weight_temp_var, axis=1, keepdims=True)  # shape: (1,             1,      n_temp_variables)
            temp_variables_bar = nontrivial_temp_var - mean_temp_var                                 # shape: (1,             n_days, n_temp_variables)
            del nontrivial_temp_var
            weight_mixed_var = weight_ks  # weight_mixed_var
            mean_mixed_var = np.nansum(nontrivial_mixed_var * weight_mixed_var, axis=(0,1), keepdims=True)  # shape: (n_individuals, n_days, n_mixed_variables)
            mixed_variables_bar = nontrivial_mixed_var - mean_mixed_var                                     # shape: (n_individuals, n_days, n_mixed_variables)
            del nontrivial_mixed_var


            # Intervals to place the variables in the matrices
            I = slice(0, n_ind_variables)
            T = slice(n_ind_variables, n_ind_variables+n_temp_variables)
            M = slice(n_exo_var-n_mixed_variables, n_exo_var)

            X_bar = X_bar[:,:,:,nax]         # shape: (n_individuals, n_days, dim, 1)
            weight_ks = weight_ks[:,:,:,nax] # shape: (n_individuals, n_days, 1,   1)

            sigma_X_Y = np.zeros((dim, n_exo_var))
            sigma_X_Y[:,I] = np.nansum(X_bar * weight_ks *   ind_variables_bar[:,:,nax,:], axis=(0,1))
            sigma_X_Y[:,T] = np.nansum(X_bar * weight_ks *  temp_variables_bar[:,:,nax,:], axis=(0,1))
            sigma_X_Y[:,M] = np.nansum(X_bar * weight_ks * mixed_variables_bar[:,:,nax,:], axis=(0,1))

            sigma_Y_Y = np.zeros((n_exo_var, n_exo_var))
            sigma_Y_Y[I,I] = half_computed_covar(ind_variables_bar, weight_ks)
            sigma_Y_Y[I,T] = np.nansum(  ind_variables_bar[:,:,:,nax] * weight_ks *  temp_variables_bar[:,:,nax,:], axis=(0,1))
            sigma_Y_Y[I,M] = np.nansum(  ind_variables_bar[:,:,:,nax] * weight_ks * mixed_variables_bar[:,:,nax,:], axis=(0,1))
            sigma_Y_Y[T,I] = sigma_Y_Y[I,T].T
            sigma_Y_Y[T,T] = half_computed_covar(temp_variables_bar, weight_ks)
            sigma_Y_Y[T,M] = np.nansum( temp_variables_bar[:,:,:,nax] * weight_ks * mixed_variables_bar[:,:,nax,:], axis=(0,1))
            sigma_Y_Y[M,I] = sigma_Y_Y[I,M].T
            sigma_Y_Y[M,T] = sigma_Y_Y[T,M].T
            sigma_Y_Y[M,M] = half_computed_covar(mixed_variables_bar, weight_ks)

            inv_sigma_Y_Y = np.linalg.pinv(sigma_Y_Y, rcond=1e-8, hermitian=True) # shape: (n_exo_var, n_exo_var)
            this_cluster_alphabetagamma = inv_sigma_Y_Y @ sigma_X_Y.T       # shape: (n_exo_var, dim)
                # we assume no NaN here.

            alpha[:,:,k,s] = this_cluster_alphabetagamma[:n_ind_variables,:]                            # shape: (n_ind_variables,  dim)
            beta[:,:,k,s]  = this_cluster_alphabetagamma[n_ind_variables:n_exo_var-n_mixed_variables,:] # shape: (n_temp_variables, dim)
            gamma[:,:,k,s] = this_cluster_alphabetagamma[n_exo_var-n_mixed_variables:,:]

            m[:,k,s] = mean_X - mean_ind_var @ alpha[:,:,k,s] - mean_temp_var @ beta[:,:,k,s] - mean_mixed_var @ gamma[:,:,k,s]
    return m, alpha, beta, gamma


#%%



if __name__ == "__main__":
    #          testing  explain_contributions_exogenous()

    # if we craft an artificial dataset using only the contributions of some exogenous
    # variables (i.e., without noise), we should be able to recompute these contributions
    # exactly from the dataset

                    # one cluster
    n_ind_variables   = 4
    n_temp_variables  = 3
    n_mixed_variables = 2
    n_individuals = 300
    n_days = 100
    dim = 12

    theoretical_alpha = np.random.randn(n_ind_variables,   dim)
    theoretical_beta  = np.random.randn(n_temp_variables,  dim)
    theoretical_gamma = np.random.randn(n_mixed_variables, dim)
    theoretical_m     = np.random.randn(dim)
    ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
    temp_variables  = np.random.randn(   1,          n_days, n_temp_variables)
    mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)

    X = theoretical_m.reshape(1,1,-1) +\
            (ind_variables   @ theoretical_alpha)  +\
            (temp_variables  @ theoretical_beta )  +\
            (mixed_variables @ theoretical_gamma)

    r = np.ones((n_individuals, n_days, 1, 1))
    variables = (ind_variables, temp_variables, mixed_variables)
    m, alpha, beta, gamma = explain_contributions_exogenous(X, variables, np.log(r))
    assert np.allclose(      m[:,0,0], theoretical_m,     1e-3, 1e-4)
    assert np.allclose(alpha[:,:,0,0], theoretical_alpha, 1e-3, 1e-4)
    assert np.allclose( beta[:,:,0,0], theoretical_beta,  1e-3, 1e-4)
    assert np.allclose(gamma[:,:,0,0], theoretical_gamma, 1e-3, 1e-4)
        # we needed to bump the tolerance to this point to get a reliable success
        #    i.e. relative tolerance is 100 times highter than its default value



                    # seven clusters, responsibilities can change irregularly
    n_clusters = 7

    r = np.zeros((n_individuals, n_days, n_clusters, 1))
    clusters = np.random.randint(0, n_clusters, size=(n_individuals, n_days))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r[i,j,clusters[i,j],0] = 1
    epsilon = 1e-15
    log_r = np.log(np.clip(r, epsilon, 1-(n_clusters-1)*epsilon))

    theoretical_alpha = np.random.randn(n_ind_variables,   dim, n_clusters, 1)
    theoretical_beta  = np.random.randn(n_temp_variables,  dim, n_clusters, 1)
    theoretical_gamma = np.random.randn(n_mixed_variables, dim, n_clusters, 1)
    theoretical_m     = np.random.randn(                  dim, n_clusters, 1)
    ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
    temp_variables  = np.random.randn(   1,          n_days, n_temp_variables)
    mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)

    X = np.zeros((n_individuals, n_days, dim))
    for k in range(n_clusters):
        X += r[:,:,k:k+1,0] * (  theoretical_m[:,k,0].reshape(1,1,-1) +\
                              (ind_variables   @ theoretical_alpha[:,:,k,0])   +\
                              (temp_variables  @ theoretical_beta[ :,:,k,0])   +\
                              (mixed_variables @ theoretical_gamma[:,:,k,0])    )


    variables = (ind_variables, temp_variables, mixed_variables)
    m, alpha, beta, gamma = explain_contributions_exogenous(X, variables, log_r)

    X_recomputed = np.zeros((n_individuals, n_days, dim))
    for k in range(n_clusters):
        X_recomputed += r[:,:,k:k+1,0] * (m[:,k,0].reshape(1,1,-1) +\
                            (ind_variables   @ alpha[:,:,k,0])  +\
                            (temp_variables  @ beta[ :,:,k,0])  +\
                            (mixed_variables @ gamma[:,:,k,0])   )


    # print(X - X_recomputed)

    assert np.allclose(    X, X_recomputed,      1e-2, 1e-3)

    assert np.allclose(    m, theoretical_m,     1e-3, 1e-4)
    assert np.allclose(alpha, theoretical_alpha, 1e-3, 1e-4)
    assert np.allclose( beta, theoretical_beta,  1e-3, 1e-4)
    assert np.allclose(gamma, theoretical_gamma, 1e-3, 1e-4)




    # Comparison with sklearn implementation

    m_sklearn, alpha_sklearn, beta_sklearn, gamma_sklearn = sklearn_explain_contributions_exogenous(X, variables, log_r)

    assert np.allclose(    m, m_sklearn,     1e-3, 1e-4)
    assert np.allclose(alpha, alpha_sklearn, 1e-3, 1e-4)
    assert np.allclose( beta, beta_sklearn,  1e-3, 1e-4)
    assert np.allclose(gamma, gamma_sklearn, 1e-3, 1e-4)




#%%
                    # several segments and segments
    n_clusters = 5
    n_trials = 5
    n_segments = 4
    uncertainties_array = np.linspace(0, 1/(n_clusters*n_segments), 21)
    errors_mean = np.zeros((uncertainties_array.shape[0], 4))
    errors_std  = np.zeros((uncertainties_array.shape[0], 4))
    for i_u, uncertainty in enumerate(uncertainties_array):
        print(f"{uncertainty:.3f}")
        this_uncertainty_errors = []
        for _ in range(n_trials):
            theoretical_alpha = np.random.randn(n_ind_variables,   dim, n_clusters, n_segments)
            theoretical_beta  = np.random.randn(n_temp_variables,  dim, n_clusters, n_segments)
            theoretical_gamma = np.random.randn(n_mixed_variables, dim, n_clusters, n_segments)
            theoretical_m     = np.random.randn(                   dim, n_clusters, n_segments)
            ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
            temp_variables  = np.random.randn(1,             n_days, n_temp_variables)
            mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)

            # hard_assignment
            hard_r = np.zeros((n_individuals, n_days, n_clusters, n_segments))
            clusters = np.random.randint(0, n_clusters, size=(n_individuals, n_days))
            segments = np.random.randint(0, n_segments, size=(n_individuals, n_days))
            for i in range(r.shape[0]):
                for j in range(r.shape[1]):
                    hard_r[i,j,clusters[i,j],segments[i,j]] = 1

            X = np.zeros((n_individuals, n_days, dim))
            for c in range(n_clusters):
                for s in range(n_segments):
                    X += hard_r[:,:,c:c+1,s] * (  theoretical_m[:,c,s].reshape(1,1,-1) +\
                                          (ind_variables   @ theoretical_alpha[:,:,c,s])  +\
                                          (temp_variables  @ theoretical_beta[ :,:,c,s])  +\
                                          (mixed_variables @ theoretical_gamma[:,:,c,s])   )
            X += np.random.randn(*X.shape)  #  Adding unit Gaussian noise

            uncertain_r = (1-uncertainty) * hard_r + uncertainty
            log_uncertain_r = np.log(np.clip(uncertain_r, epsilon, 1-(n_clusters*n_segments-1)*epsilon))
                # uncertainty can be zero

            variables = (ind_variables, temp_variables, mixed_variables)
            m, alpha, beta, gamma = explain_contributions_exogenous(X, variables, log_uncertain_r)

            alpha_error = np.mean(np.abs(alpha - theoretical_alpha))
            beta_error  = np.mean(np.abs(beta  - theoretical_beta))
            gamma_error = np.mean(np.abs(gamma - theoretical_gamma))
            m_error     = np.mean(np.abs( m    - theoretical_m))
            this_uncertainty_errors.append((alpha_error, beta_error, gamma_error, m_error))

        errors_mean[i_u,0] = np.mean([alpha_err for (alpha_err, _, _, _) in this_uncertainty_errors])
        errors_mean[i_u,1] = np.mean([ beta_err for (_, beta_err, _, _)  in this_uncertainty_errors])
        errors_mean[i_u,2] = np.mean([gamma_err for (_, _, gamma_err, _) in this_uncertainty_errors])
        errors_mean[i_u,3] = np.mean([    m_err for (_, _, _, m_err)     in this_uncertainty_errors])
        errors_std[ i_u,0] = np.std( [alpha_err for (alpha_err, _, _, _) in this_uncertainty_errors])
        errors_std[ i_u,1] = np.std( [ beta_err for (_, beta_err, _, _)  in this_uncertainty_errors])
        errors_std[ i_u,2] = np.std( [gamma_err for (_, _, gamma_err, _) in this_uncertainty_errors])
        errors_std[ i_u,3] = np.std( [    m_err for (_, _, _, m_err)     in this_uncertainty_errors])




    plt.figure()
    plt.errorbar(uncertainties_array, errors_mean[:,0], yerr=errors_std[:,0], label=r'$\alpha$')
    plt.errorbar(uncertainties_array, errors_mean[:,1], yerr=errors_std[:,1], label=r'$\beta$')
    plt.errorbar(uncertainties_array, errors_mean[:,2], yerr=errors_std[:,2], label=r'$\gamma$')
    plt.errorbar(uncertainties_array, errors_mean[:,3], yerr=errors_std[:,3], label=r'$m$')
    plt.legend()
    plt.title("Estimation of the influence of exogenous variables:\nmean +/- std $L_1$ error vs.\n uncertainty on cluster/segment labels")
    plt.grid(True)
    plt.ylabel(r"mean (on parameters) $L_1$ error ")
    plt.xlabel(f"uncertainty on the (cluster x segment) label ({n_clusters} clusters, {n_segments} segments)")
    plt.show()







#%%


    n_individuals = 40
    n_days = 40

    nan_proportion_array = np.linspace(0, 0.9, 19)
    errors_mean = np.zeros((nan_proportion_array.shape[0], 4))
    errors_std  = np.zeros((nan_proportion_array.shape[0], 4))
    for i_u, nanprop in enumerate(nan_proportion_array):
        this_error_list = []
        print(f"{nanprop:.2f}")

        for _ in range(n_trials):
            theoretical_alpha = np.random.randn(n_ind_variables,  dim, n_clusters, n_segments)
            theoretical_beta  = np.random.randn(n_temp_variables, dim, n_clusters, n_segments)
            theoretical_gamma = np.random.randn(n_mixed_variables, dim, n_clusters, n_segments)
            theoretical_m     = np.random.randn(                   dim, n_clusters, n_segments)
            ind_variables   = np.random.randn(n_individuals, 1,      n_ind_variables)
            temp_variables  = np.random.randn(1,             n_days, n_temp_variables)
            mixed_variables = np.random.randn(n_individuals, n_days, n_mixed_variables)


            # hard_assignment
            hard_r = np.zeros((n_individuals, n_days, n_clusters, n_segments))
            clusters = np.random.randint(0, n_clusters, size=(n_individuals, n_days))
            segments = np.random.randint(0, n_segments, size=(n_individuals, n_days))
            for i in range(n_individuals):
                for j in range(n_days):
                    hard_r[i,j,clusters[i,j],segments[i,j]] = 1
            log_r = np.log(np.clip(hard_r, epsilon, 1-(n_clusters*n_segments-1)*epsilon))

            X = np.zeros((n_individuals, n_days, dim))
            for c in range(n_clusters):
                for s in range(n_segments):
                    X += hard_r[:,:,c:c+1,s] * (  theoretical_m[:,c,s].reshape(1,1,-1) +\
                                          (ind_variables   @ theoretical_alpha[:,:,c,s])  +\
                                          (temp_variables  @ theoretical_beta[ :,:,c,s])  +\
                                          (mixed_variables @ theoretical_gamma[:,:,c,s]) )
            X += np.random.randn(*X.shape)  #  Adding unit Gaussian noise

            # add nans
            isnan = np.random.uniform(0,1, (n_individuals, n_days, 1)) < nanprop

            mask = np.ones((n_individuals, n_days, 1))
            mask[isnan] = np.nan
            X *= mask

            variables = (ind_variables, temp_variables, mixed_variables)
            m, alpha, beta, gamma = explain_contributions_exogenous(X, variables, log_r)

            alpha_error = np.mean(np.abs(alpha - theoretical_alpha))
            beta_error  = np.mean(np.abs(beta  - theoretical_beta))
            gamma_error = np.mean(np.abs(gamma - theoretical_gamma))
            m_error     = np.mean(np.abs( m    - theoretical_m))
            this_error_list.append((alpha_error, beta_error, gamma_error, m_error))


        errors_mean[i_u,0] = np.mean([alpha_err for (alpha_err, _, _, _) in this_error_list])
        errors_mean[i_u,1] = np.mean([ beta_err for (_, beta_err, _, _)  in this_error_list])
        errors_mean[i_u,2] = np.mean([gamma_err for (_, _, gamma_err, _) in this_error_list])
        errors_mean[i_u,3] = np.mean([    m_err for (_, _, _, m_err)     in this_error_list])
        errors_std[ i_u,0] = np.std( [alpha_err for (alpha_err, _, _, _) in this_error_list])
        errors_std[ i_u,1] = np.std( [ beta_err for (_, beta_err, _, _)  in this_error_list])
        errors_std[ i_u,2] = np.std( [gamma_err for (_, _, gamma_err, _) in this_error_list])
        errors_std[ i_u,3] = np.std( [    m_err for (_, _, _, m_err)     in this_error_list])



    plt.figure()
    plt.errorbar(nan_proportion_array, errors_mean[:,0], yerr=errors_std[:,0], label=r'$\alpha$')
    plt.errorbar(nan_proportion_array, errors_mean[:,1], yerr=errors_std[:,1], label=r'$\beta$')
    plt.errorbar(nan_proportion_array, errors_mean[:,2], yerr=errors_std[:,2], label=r'$\gamma$')
    plt.errorbar(nan_proportion_array, errors_mean[:,3], yerr=errors_std[:,3], label=r'$m$')
    plt.legend()
    plt.title("Estimation of the influence of exogenous variables:\nmean +/- std $L_1$ error vs.\n proportion of NaNs in X")
    plt.grid(True)
    plt.ylabel(r"mean (on parameters) $L_1$ error ")
    plt.xlabel("Proportion of NaN values")
    plt.show()

