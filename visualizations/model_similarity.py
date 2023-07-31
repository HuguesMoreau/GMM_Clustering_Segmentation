import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score


def predict_clusters(model, X, variables):
    """
    Parameters
    ----------
    model: a GMM_segmentation instance
    X: np array with shape (n_individuals, n_days, dim)
    variables: optional, triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None
    Returns
    -------
    preds: np array of ints with shape (n_individuals, n_days)
        Each value is in [0, n_clusters*n_segments[
        May contain NaNs
    """
    n_individuals, n_days, _ = X.shape

    log_r, _ = model.E_step(X, variables) # shape: (n_individuals, n_days, n_clusters, n_segments)
    log_r = log_r.reshape(n_individuals * n_days, -1)
    log_r[np.isnan(log_r)] = -np.inf  # argmax([nan, 100])  returns 0
    preds = np.argmax(log_r, axis=1)
    return preds








def plot_model_agreement_from_predictions(pred_dict):
    """
    Compute the expectation, for all samples in some origin segment and cluster, of the
    log-likelihood of this element to belong in each other destination cluster.
    This only takes into account the Gaussian and exogenous variables, and ignores the
    mixture weights and segment locations.

    Parameters
    ----------
    pred_dict: a dictionnary with
        keys: strings (model names)
        values: list of np arrays with shape (n_individuals * n_days)

    Returns
    -------
    None

    """

    model_names_list = list(pred_dict.keys())  # we need an ordering
    n_model_types = len(pred_dict)
    mean_scores = np.zeros((n_model_types, n_model_types))
    std_scores  = np.zeros((n_model_types, n_model_types))

    for i_model_type_1, model_name_1 in enumerate(model_names_list):
        for i_model_type_2, model_name_2 in enumerate(model_names_list[:i_model_type_1+1]):

            scores_list = []

            for i_pred_1, pred_1 in enumerate(pred_dict[model_name_1]):
                end_i_2 = len(pred_dict[model_name_2]) if model_name_2 != model_name_1 else i_pred_1
                # i_pred_1 instead of i_pred_1+1 because we do not compute the similarity of a model with itself :)
                for i_pred_2, pred_2 in enumerate(pred_dict[model_name_2][:end_i_2]):
                    this_ari = adjusted_rand_score(pred_dict[model_name_1][i_pred_1],
                                                   pred_dict[model_name_2][i_pred_2])
                    scores_list.append(this_ari)

            mean = np.mean(scores_list)
            std  =  np.std(scores_list)
            mean_scores[i_model_type_1, i_model_type_2] = mean
            mean_scores[i_model_type_2, i_model_type_1] = mean
            std_scores[ i_model_type_1, i_model_type_2] = std
            std_scores[ i_model_type_2, i_model_type_1] = std


    plt.figure(figsize=(12, 12))
    min_score = mean_scores.min()
    max_score = mean_scores.max()
    plt.imshow(mean_scores, vmin=min_score, vmax=max_score) # imshow swaps axes
    plt.colorbar()
    plt.xticks(np.arange(n_model_types), model_names_list, rotation=-45, ha='left')
    plt.yticks(np.arange(n_model_types), model_names_list)


    for i_model_type_1, model_name_1 in enumerate(model_names_list):
        for i_model_type_2, model_name_2 in enumerate(model_names_list):


            this_mean = mean_scores[i_model_type_1, i_model_type_2]
            this_std  = std_scores[ i_model_type_2, i_model_type_1]
            ARI_string = f"{this_mean:.3f} \n ±{this_std:.3f}"

            textcolor = "k" if this_mean > (min_score+max_score)/2 else "w"
            plt.text(x=i_model_type_1, y=i_model_type_2, # imshow swaps the y axis
                      s=ARI_string, ha="center", va="center", c=textcolor, fontsize=9)











def plot_model_agreement(model_dict, X, variables):
    """
    Compute the expectation, for all samples in some origin segment and cluster, of the
    log-likelihood of this element to belong in each other destination cluster.
    This only takes into account the Gaussian and exogenous variables, and ignores the
    mixture weights and segment locations.

    Parameters
    ----------
    model_dict: a dictionnary with
        keys: strings (model names)
        values: list of models
    X: np array with shape (n_individuals, n_days, dim)
    variables: optional, triple of:
        ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)   or None
        temp_variables:  np array with shape (1,             n_days, n_temp_variables)  or None
        mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables) or None

    Returns
    -------
    None

    """
    n_individuals, n_days, _ = X.shape


    # forst step: record all

    all_predictions = {}
    for model_name, model_list in model_dict.items():
        all_predictions[model_name] = []

        for model in model_list:
            all_predictions[model_name].append(predict_clusters(model, X, variables))

    plot_model_agreement_from_predictions(all_predictions)






#%%
if __name__ == "__main__":
    from models.main_model import GMM_segmentation
    from visualizations.clustering_results import plot_mean_covar


    n_individuals, n_days = 1000, 1



    model_1 = GMM_segmentation(dim=2, n_clusters=2, n_segments=1, covariance_type="diag")
    model_2 = GMM_segmentation(dim=2, n_clusters=2, n_segments=1, covariance_type="diag")

    # Model 1
    model_1.m[:,0,0] = np.array([0,0])
    model_1.sigma[:,0,0] = np.array([1,5])
    model_1.m[:,1,0] = np.array([15,30])
    model_1.sigma[:,1,0] = np.array([15,5])

    # Model 2
    model_2.m[:,0,0] = np.array([30,0])
    model_2.sigma[:,0,0] = np.array([10,1])
    model_2.m[:,1,0] = np.array([15,30])
    model_2.sigma[:,1,0] = np.array([2,2])

    colors = np.zeros((3,2,2))
    colors[:,0,0] = np.array([0.00, 0.87, 0.43])
    colors[:,0,1] = np.array([0.65, 0.86, 0.07])
    colors[:,1,0] = np.array([0.71, 0.16, 0.38])
    colors[:,1,1] = np.array([0.56, 0.56, 0.87])

    X = np.array([[[15, 15]]]) + np.random.randn(n_individuals, n_days, 2) * 7

    ind_variables   = np.zeros((n_individuals, 1,      0), dtype=float)
    temp_variables  = np.zeros((1,             n_days, 0), dtype=float)
    mixed_variables = np.zeros((n_individuals, n_days, 0), dtype=float)
    variables = (ind_variables, temp_variables, mixed_variables)

    # Plotting the models
    plt.figure(figsize=(10, 10))
    for i_model, model in enumerate([model_1, model_2]):
        for k in [0,1]:
            color = list(colors[:,k,i_model])

            plot_mean_covar(mu=model.m[:,k,0], cov=np.diag(model.sigma[:,k,0]), c=color,
                            mean_label=f'model {i_model+1}, cluster {k+1}')

    # plotting the data
    pred_1 = predict_clusters(model_1, X, variables)
    pred_2 = predict_clusters(model_2, X, variables)
    color_edge = colors[:,:,0][:,pred_1].T   #  model 1, shape: (n_individuals * n_days, 3)
    color_face = colors[:,:,1][:,pred_2].T   #  model 1, shape: (n_individuals * n_days, 3)
    plt.scatter(X[:,:,0].reshape(-1), X[:,:,1].reshape(-1),
                edgecolor=color_edge, facecolor=color_face,
                s=40, marker="d", label="data")
    plt.legend()
    plot_model_agreement({"model_1":[model_1, model_1], "model_2":[model_2, model_2]},
                          X, variables)




