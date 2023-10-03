import datetime
from copy import deepcopy
import time
import random
import numpy as np


from models.main_model import GMM_segmentation
from models.SegClust import SegClust

from preprocessing.IdFM.load_data import load_data as load_data_IdFM


random.seed(0)
np.random.seed(0)


#%%






def evaluation_process(Y_train, variables_train, Y_val, variables_val, n_clusters, n_segments,
                       evaluation_fn, **kwargs):
    """
    Parameters
    ----------
    Y_train: np array with shape (n_individuals_train, n_days_train, dim)
    variables_train: triple of:
        ind_variables:   np array with shape (n_individuals_train, 1,            n_ind_variables)
        temp_variables:  np array with shape (1,                   n_days_train, n_temp_variables)
        mixed_variables: np array with shape (n_individuals_train, n_days_train, n_mixed_variables)
    Y_val: np array with shape (n_individuals_val, n_days_val, dim)
    variables_val: the descritption for this variable is left as an exercise to the reader
    n_clusters: int
    n_segments: int
    evaluation_fn: function with signature (model, Y, variables, r) => float
    **kwargs : arguments for the model creation
        (there is no *args)

    Returns
    -------
    results: dict with
        keys: "Reg > Clust > Seg", "(Clust+Reg) > (Seg+Reg)", etc.
        values: outputs of evaluation_fn
    times: same dict, except that the values are the times spent for each method
    """

    results, times  = {}, {}
    n_ind_variables   = variables_train[0].shape[2]
    n_temp_variables  = variables_train[1].shape[2]
    n_mixed_variables = variables_train[2].shape[2]
    dim = Y_train.shape[2]
    empty_variables = None



    # =============================================================================
    #                        Reg > Clust > Seg
    # =============================================================================
    # note that simple linear regression, clustering, or segmentation are three particular cases of our model

    # Regression
    previous_time = time.time()
    model_regression = GMM_segmentation(dim=dim, n_clusters=1, n_segments=1,
                                        n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                                        **kwargs)
    model_regression.EM(Y_train, variables_train,
             init_method=('parameters', ("random", "uniform")), print_every=0, patience=0)
    Y_train_corrected = model_regression.get_residuals(Y_train, variables_train) # shape: (n_individuals, n_days, dim, 1, 1)
    Y_val_corrected   = model_regression.get_residuals(Y_val,   variables_val)
    Y_train_corrected = Y_train_corrected[:,:,:,0,0]
    Y_val_corrected   = Y_val_corrected[:,:,:,0,0]
    time_regression = time.time() - previous_time  # we will re-use the regression later, so we will add the regression time

    # Clustering
    model_clustering = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=1,
                                   n_variables=(0, 0, 0),
                                  **kwargs)
    model_clustering.EM(Y_train_corrected, empty_variables,
                        init_method=('parameters', ("random", "uniform")), print_every=0)
    log_r_train_clusters, _ = model_clustering.E_step(Y_train_corrected, empty_variables)   # shape: (n_individuals, n_days, n_clusters, 1)
    log_r_val_clusters,   _ = model_clustering.E_step(  Y_val_corrected, empty_variables)   # shape: (n_individuals, n_days, n_clusters, 1)
    clusters_train = np.argmax(np.sum(log_r_train_clusters[...,0], axis=1), axis=1)   # shape: (n_individuals,)
    clusters_val   = np.argmax(np.sum(log_r_val_clusters[  ...,0], axis=1), axis=1)   # shape: (n_individuals,)



    final_model = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=n_segments,
                                   n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                                  **kwargs)
    final_model.CEM = False
    final_model.pi = model_clustering.pi
    final_model.alpha = final_model.alpha*0 + model_regression.alpha  # broadcasting the contribution to all cluster and segments
    final_model.beta  = final_model.beta *0 + model_regression.beta

    log_r_val = np.zeros((log_r_val_clusters.shape[0], log_r_val_clusters.shape[1], log_r_val_clusters.shape[2], n_segments)) -np.inf
    for k in range(n_clusters):
        these_inds_train = (clusters_train == k)
        these_inds_val   = (clusters_val   == k)
        if these_inds_train.sum() > 0:
            model_segmentation = GMM_segmentation(dim=dim, n_clusters=1, n_segments=n_segments,
                                      n_variables=(0, 0, 0),
                                     **kwargs)
            model_segmentation.EM(Y_train_corrected[these_inds_train,...], empty_variables,
                      init_method=('parameters', ("random", "uniform")), print_every=0)

            final_model.m[:,k,:] = model_segmentation.m[:,0,:]
            final_model.u[ k, :] = model_segmentation.u[0,:]
            final_model.v[ k, :] = model_segmentation.v[0,:]

            if these_inds_val.sum() > 0:
                log_r_val_segments, _ = model_segmentation.E_step(Y_val_corrected[these_inds_val,...], empty_variables)
                        # shape: (n_these_inds, n_days, 1, n_segments)
                log_r_val[these_inds_val,:,k:k+1,:] = log_r_val_segments


    results["Reg > Clust > Seg"] = evaluation_fn(final_model, Y_val, variables_val, log_r_val)
    times["Reg > Clust > Seg"] = (time.time() - previous_time)







    # =============================================================================
    #                       (Clust+Reg) > (Seg+Reg)
    # =============================================================================

    # Clustering
    previous_time = time.time()
    model_clustering = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=1,
                                          n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                             **kwargs)
    model_clustering.EM(Y_train, variables_train,
              init_method=('parameters', ("random", "uniform")), print_every=0)
    log_r_train_clusters, _ = model_clustering.E_step(Y_train, variables_train)   # shape: (n_individuals, n_days, n_clusters, 1)
    log_r_val_clusters,   _ = model_clustering.E_step(  Y_val, variables_val)   # shape: (n_individuals, n_days, n_clusters, 1)
    clusters_train = np.argmax(np.sum(log_r_train_clusters[...,0], axis=1), axis=1)   # shape: (n_individuals,)
    clusters_val   = np.argmax(np.sum(log_r_val_clusters[  ...,0], axis=1), axis=1)   # shape: (n_individuals,)



    final_model = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=n_segments,
                                   n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                                  **kwargs)
    final_model.CEM = False
    final_model.pi = model_clustering.pi

    log_r_val = np.zeros((log_r_val_clusters.shape[0], log_r_val_clusters.shape[1], log_r_val_clusters.shape[2], n_segments)) -np.inf
    for k in range(n_clusters):
        these_inds_train = (clusters_train == k)
        these_inds_val   = (clusters_val   == k)
        if these_inds_train.sum() > 0:
            model_segmentation = GMM_segmentation(dim=dim, n_clusters=1, n_segments=n_segments,
                                           n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                                          **kwargs)
            these_variables_train = (variables_train[0][these_inds_train,:,:], variables_train[1], variables_train[2][these_inds_train,:,:])
            model_segmentation.EM(Y_train[these_inds_train,...], these_variables_train,
                      init_method=('parameters', ("random", "uniform")), print_every=0)

            final_model.m[:,k,:] = model_segmentation.m[:,0,:]
            final_model.u[ k, :] = model_segmentation.u[0,:]
            final_model.v[ k, :] = model_segmentation.v[0,:]

            final_model.alpha[:,:,k,:] = model_segmentation.alpha[:,:,0,:]
            final_model.beta[ :,:,k,:] = model_segmentation.beta[ :,:,0,:]

            if these_inds_val.sum() > 0:
                these_variables_val =  (variables_val[0][these_inds_val,:,:], variables_val[1], variables_val[2][these_inds_val,:,:])
                log_r_val_segments, _ = model_segmentation.E_step(Y_val[these_inds_val,...], these_variables_val)
                        # shape: (n_these_inds, n_days, 1, n_segments)
                log_r_val[these_inds_val,:,k:k+1,:] = log_r_val_segments

    results["(Clust+Reg) > (Seg+Reg)"]  = evaluation_fn(final_model, Y_val, variables_val, log_r_val)
    times["(Clust+Reg) > (Seg+Reg)"] = (time.time() - previous_time)







    # =============================================================================
    #                        Reg > Clust+Seg
    # =============================================================================
     # reuse the model_regression from previously
    previous_time = time.time()
    model_clustseg = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=n_segments,
                             n_variables=(0, 0, 0),
                            **kwargs)

    model_clustseg.EM(Y_train_corrected, empty_variables,
             init_method=('parameters', ("random", "uniform")), print_every=0)
    log_r_train, _ = model_clustseg.E_step(Y_train_corrected, empty_variables)
    log_r_val,   _ = model_clustseg.E_step(Y_val_corrected,   empty_variables)


    final_model = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=n_segments,
                                   n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                                  **kwargs)
    final_model.CEM = False
    final_model.alpha = final_model.alpha*0 + model_regression.alpha  # broadcasting the contribution to all cluster and segments
    final_model.beta  = final_model.beta *0 + model_regression.beta
    final_model.gamma = final_model.gamma*0 + model_regression.gamma

    final_model.m     = model_clustseg.m + model_regression.m

    final_model.pi    = model_clustseg.pi
    final_model.u     = model_clustseg.u
    final_model.v     = model_clustseg.v
    final_model.sigma = model_clustseg.sigma


    results["Reg > Clust+Seg"] = evaluation_fn(final_model, Y_val, variables_val, log_r_val)
    times["Reg > Clust+Seg"] = (time.time() - previous_time) + time_regression  # do not forget that we need the regression here too

    assert np.isclose(evaluation_fn(final_model,    Y_train,           variables_train, log_r_train),
                      evaluation_fn(model_clustseg, Y_train_corrected, empty_variables, log_r_train),
                      atol=1e-5, rtol=1e-7)
    # assert np.isclose(evaluation_fn(final_model,    Y_val,             variables_val,   log_r_val),
    #                   evaluation_fn(model_clustseg, Y_val_corrected,   empty_variables, log_r_val),
    #                   atol=1e-5, rtol=1e-7)
        # In practice, parameter m absorbs the mean of exogenous variables, which may slightly change for ind variables.






    # =============================================================================
    #                         Clust+Seg
    # =============================================================================

    previous_time = time.time()
    model_clustseg = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=n_segments,
                             n_variables=(0, 0, 0),
                            **kwargs)

    model_clustseg.EM(Y_train, empty_variables,
             init_method=('parameters', ("random", "uniform")), print_every=0)
    log_r_train, _ = model_clustseg.E_step(Y_train, empty_variables)
    log_r_val,   _ = model_clustseg.E_step(Y_val,   empty_variables)

    results["Clust+Seg"] = evaluation_fn(model_clustseg, Y_val, empty_variables, log_r_val)
    times["Clust+Seg"] = (time.time() - previous_time)








    # =============================================================================
    #                         Clust+Seg > Reg
    # =============================================================================
    # re-use the clust+seg model from previously
    # thats why we do not update previous_time

    final_model = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=n_segments,
                                   n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                                  **kwargs)
    final_model.CEM = False

    # Fitting both regression parameters and sigma
    final_model.M_step(Y_train, variables_train, log_r_train)

    final_model.pi = model_clustseg.pi
    final_model.u  = model_clustseg.u
    final_model.v  = model_clustseg.v

    results["Clust+Seg > Reg"] = evaluation_fn(final_model, Y_val, variables_val, log_r_val)
    times["Clust+Seg > Reg"] = (time.time() - previous_time)






    # =============================================================================
    #                             Our model
    # =============================================================================
    previous_time = time.time()
    model = GMM_segmentation(dim=dim, n_clusters=n_clusters, n_segments=n_segments,
                                   n_variables=(n_ind_variables, n_temp_variables, n_mixed_variables),
                                  **kwargs)
    model.EM(Y_train, variables_train,
             init_method=('parameters', ("random", "uniform")),
             init_with_CEM=False, CEM=False, print_every=0)
    log_r_val, _ = model.E_step(Y_val, variables_val)

    results["Clust+Seg+Reg"] = evaluation_fn(model, Y_val, variables_val, log_r_val)
    times["Clust+Seg+Reg"] = (time.time() - previous_time)




    return results, times











#%%

if __name__ == "__main__":



    def evaluation_fn_real(model, Y, variables, log_r):
        return model.LogLikelihood(Y, variables)



    chosen_variables =  ["strike", "strike_2019", "lockdown_different", "school_holiday", "national_holiday",
                         "year_splines_d2_n21", "day_of_week", "time", "rail_type"]


    data, variable_names, id_names, IdFM_origin = load_data_IdFM("rail", ticket_types="merge", start_date=datetime.date(2017,1,1), sampling=None,
                                                    chosen_variables=chosen_variables, normalisation="individual",
                                                    remove_mean="none", scale='log10')

    (Y, ind_array, day_array, variables) = data[0]

    n_clusters = 5
    n_segments = 2
    n_days = data[0][0].shape[2]
    min_slope = 4.4 * (n_days/90)  # from 1% to 99% (4.4) in less than three months

    Y, _, _, exogenous_variables = data[0]  # we do not train/test split herre to be able to use cross-validation
    ind_variables, temp_variables, mixed_variables = exogenous_variables


    ind_variables, temp_variables, mixed_variables = exogenous_variables
    n_individuals, n_days, _ = Y.shape



    LL_dict = {"Reg > Clust > Seg":[],
               "(Clust+Reg) > (Seg+Reg)":[],
               "Reg > Clust+Seg":[],
               "Clust+Seg":[],
               "Clust+Seg > Reg":[],
               "SegClust_Picard":[],
               "Clust+Seg+Reg":[] }

    times_dict = deepcopy(LL_dict)


    n_splits = 5
    random_set = np.floor(np.linspace(0, n_splits, Y.shape[0], endpoint=False))
    np.random.shuffle(random_set)


    for i_trial in range(n_splits):

        val_individuals = np.isclose(random_set, i_trial)

        Y_train = Y[~val_individuals,...]
        Y_val  =  Y[ val_individuals,...]

        ind_var_train = ind_variables[~val_individuals,...]
        ind_var_val  =  ind_variables[ val_individuals,...]
        mixed_variables_train = mixed_variables[~val_individuals,...]
        mixed_variables_val   = mixed_variables[ val_individuals,...]

        variables_train = (ind_var_train, temp_variables, mixed_variables_train)
        variables_val   = (ind_var_val,   temp_variables, mixed_variables_val)

        n_individuals_train, n_days, dim = Y_train.shape

        kwargs_model = {"covariance_type":"diag", "min_slope":min_slope}

        this_LL_dict, this_time = evaluation_process(Y_train, variables_train, Y_val, variables_val, n_clusters, n_segments,
                                                     evaluation_fn=evaluation_fn_real,
                                                     **kwargs_model)


        for k in this_LL_dict.keys():
            LL_dict[k].append(this_LL_dict[k])
            times_dict[k].append(this_time[k])



        print('\n\n')
        for k, v in LL_dict.items():
            print(f"{k:25s}: LL = {np.mean(v):.3f} +/- {np.std(v):.3f}")
        print()
        for k, v in times_dict.items():
            print(f"{k:25s}: t = {np.mean(v):.3f} +/- {np.std(v):.3f} s")
        print()
        print('\n\n')


    print('\n\n')
    for k, v in LL_dict.items():
        print(f"{k:25s}: LL = {np.mean(v):.3f} +/- {np.std(v):.3f}")
    print()
    for k, v in times_dict.items():
        print(f"{k:25s}: t = {np.mean(v):.3f} +/- {np.std(v):.3f} s")
    print()
    print('\n\n')



