import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp




def plot_covariances_clusters(model, X, ind_variables, temp_variables):



    log_r, _ = model.E_step(X, ind_variables, temp_variables)
    log_r -= logsumexp(log_r, axis=2, keepdims=True)
    r = np.exp(log_r)
    del log_r

    n_individuals, n_days, dim = X.shape
    n_clusters = r.shape[2]


    pca_per_cluster = []

    for c in range(n_clusters):
        this_cluster_weight = r[:,:,c] /  np.sum(r[:,:,c])
        this_cluster_weight = this_cluster_weight[:,:,np.newaxis] #shape: (n_ind, n_days, 1)
        this_cluster_mean = this_cluster_weight * (model.m[:,c].reshape(1,1,-1) +\
                        (ind_variables  @ model.alpha[:,:,c])[:,np.newaxis,:]  +\
                        (temp_variables @ model.beta[ :,:,c])[np.newaxis,:,:]   ) # shape: (n_ind, n_days, dim)
        deviation_to_mean = X - this_cluster_mean   # shape: (n_ind, n_days, dim)

        sqrt_weight = np.sqrt(this_cluster_weight)
        deviation_to_mean *= sqrt_weight
        covariance = np.tensordot(deviation_to_mean, deviation_to_mean, axes=[[0,1],[0,1]])

        values, vectors = np.linalg.eigh(covariance)  # covariance is hermitian
        pca_per_cluster.append((values[::-1], vectors[:,::-1]))
        # pca_per_cluster.append((values, vectors))
            # the eigenvalues were soirted increasingly, we want them sorted decreasing


    plt.figure()

    for i_c in range(n_clusters):
        for j_c in range(0, i_c+1):
            if i_c == j_c:
                plt.subplot(n_clusters, n_clusters, i_c + j_c*n_clusters +1)
                values, _ = pca_per_cluster[i_c]
                plt.plot(values)
                plt.xticks(np.arange(0, dim+1, 12), [" "]*(dim//12 +1))
                plt.grid(True)

            else: # i_c > j_c
                _, vectors_i = pca_per_cluster[i_c]
                _, vectors_j = pca_per_cluster[j_c]
                scalar_products = (vectors_i.T) @ vectors_j
                angles = np.arccos(np.clip(scalar_products, -1,1))

                plt.subplot(n_clusters, n_clusters, i_c + j_c*n_clusters +1)
                plt.imshow(angles/np.pi, vmin=0, vmax=1, cmap="bwr")
                # plt.xticks([],[])
                # plt.yticks([],[])

                plt.subplot(n_clusters, n_clusters, j_c + i_c*n_clusters +1)
                plt.imshow(angles.T/np.pi, vmin=0, vmax=1, cmap="bwr")
                # plt.xticks([],[])
                # plt.yticks([],[])




















