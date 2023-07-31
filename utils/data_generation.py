import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def generate_inconsistent_data(r, m, contributions, sigma, variables, covariance_type="infer"):
    """
    Generates data where the cluster can vary on each day x individual independently from the other days.

    Parameters
    ----------
    r: np array with shape (n_samples, n_days, n_clusters, n_segments)
        the responsibilities, expectet to be either 0 or 1.
    m: np array with shape (dim, n_clusters, n_segments)
    contributions: triple of:
        alpha: np array with shape (n_ind_variables,   dim, n_clusters, n_segments)
        beta : np array with shape (n_temp_variables,  dim, n_clusters, n_segments)
        gamma: np array with shape (n_mixed_variables, dim, n_clusters, n_segments)
    sigma: np array with variable shape, must match the covariance_type argument
    variables: optional, triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None
    covariance_type: string, one of 'full', 'tied', 'diag', 'spherical', 'infer' (default)
        This argument is mainly here in the improbable case where dim == n_clusters

    Returns
    -------
    Y: np array with shape (n_individuals, n_days, dim)
    """

    ind_variables, temp_variables, mixed_variables = variables
    alpha, beta, gamma = contributions

    n_individuals, n_days, n_clusters, n_segments = r.shape  # note that in this function, clusters and segments are interchangeable
    dim = m.shape[0]
    if covariance_type == "infer":
        if sigma.shape == (dim, dim, n_clusters, n_segments):
            covariance_type = "full"
        elif sigma.shape == (n_clusters,n_segments):
            covariance_type = "spherical"
        elif sigma.shape == (dim, n_clusters, n_segments):
            covariance_type = "diag"
        elif sigma.shape == (dim, dim):
            covariance_type = "tied"
        else :
            raise ValueError("Unrecognized covariance")

    Y = np.zeros((n_individuals, n_days, dim), dtype=np.float32)
    for k in range(n_clusters):
        for s in range(n_segments):
            this_cluster_mean  = m[:,k,s].reshape(1,1,-1) +\
                                 (ind_variables   @ alpha[:,:,k,s]) +\
                                 (temp_variables  @ beta[ :,:,k,s]) +\
                                 (mixed_variables @ gamma[:,:,k,s])
            if covariance_type == "full":
                sqrt_sigma = sqrtm(sigma[:,:,k,s])
                noise = np.random.randn(*Y.shape) @ sqrt_sigma
            elif covariance_type == "tied":
                noise = np.random.randn(*Y.shape) @ sqrtm(sigma)
            elif covariance_type == "diag":
                noise = np.random.randn(*Y.shape) * np.sqrt(sigma[:,k,s])[np.newaxis,np.newaxis,:]
            elif covariance_type == "spherical":
                noise = np.random.randn(*Y.shape) * np.sqrt(sigma[k,s])
                # In all cases, noise has the same shape as Y
            this_cluster_realization = this_cluster_mean + noise
            Y += r[:,:,k,s][:,:,np.newaxis] * this_cluster_realization

    return Y







def generate_data(pi, u, v, m, contributions, sigma, variables, covariance_type="infer"):
    """
    Generates data according to the definitive model
    /!\ Contrary to the previous function, this method fgenerates both the hidden data (responsibilities)
    and the visible data

    Parameters
    ----------
    pi: np array with shape (n_clusters,), sums to one
    u: np array with shape (n_clusters, n_segments)
    v: np array with shape (n_clusters, n_segments)
    m: np array with shape (dim, n_clusters, n_segments)
    contributions: triple of:
        alpha: np array with shape (n_ind_variables,   dim, n_clusters, n_segments)
        beta : np array with shape (n_temp_variables,  dim, n_clusters, n_segments)
        gamma: np array with shape (n_mixed_variables, dim, n_clusters, n_segments)
    sigma: np array with variable shape, must match the covariance_type argument
    variables: optional, triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None
    covariance_type: string, one of 'full', 'tied', 'diag', 'spherical', 'infer' (default)
        This argument is mainly here in the improbable case where dim == n_clusters or n_segments

    Returns
    -------
    Y: np array with shape (n_individuals, n_days, dim)
    r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
    """

    _, _, mixed_variables = variables
    n_individuals, n_days, _ =  mixed_variables.shape
    dim, n_clusters, n_segments = m.shape


    # =============================================================================
    #    Step 1: computation of the responsibilities:
    # =============================================================================
    # To choose among K choices with unbalanced probabilities, generate K floats between 0 and 1,
    # multiply each of them by their respective probability, and choose the highest one.

    # cluster assigment
    random_uniform_per_cluster = np.random.uniform(0,1, size=(n_individuals, n_clusters))
    random_weighted_per_cluster = random_uniform_per_cluster * pi.reshape(1,-1)
    cluster_assignment = np.argmax(random_weighted_per_cluster, axis=1)

    # segment assignment
    r  = np.zeros((n_individuals, n_days, n_clusters, n_segments))
    t = np.linspace(0, 1, n_days).reshape(-1, 1)
    for k in range(n_clusters):
        selected_individuals = (cluster_assignment == k)

        this_u, this_v = u[k,:].reshape(1, -1), v[k,:].reshape(1, -1)
        probabilities = np.exp(this_u*t + this_v)      # shape: (n_days, n_segments)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True) # shape: (n_days, n_segments)

        random_uniform_per_segment = np.random.uniform(0,1, size=(selected_individuals.sum(), n_days, n_segments))  # shape: (n_selected_ind, n_days, n_segments)
        random_weighted_per_segment = random_uniform_per_segment * probabilities[np.newaxis,:,:]                    # shape: (n_selected_ind, n_days, n_segments)
        segment_assignment = np.argmax(random_weighted_per_segment, axis=2)   # shape: (n_selected_ind, n_days)

        segment_index = np.arange(n_segments).reshape(1, 1, -1)  # shape: (1, 1, n_segments)
        r[selected_individuals,:,k,:] = (segment_assignment[:,:,np.newaxis] == segment_index)

    # =============================================================================
    #    Step 2: computation of the actual data
    # =============================================================================
    Y = generate_inconsistent_data(r, m, contributions, sigma, variables, covariance_type)
     # if the segments in r are time-consistent, the clusters in x will be too :)

    return Y, r






















