"""
This file requires that the scriptn_clusters_selection.py to have recorded some results in
results/n_clusters_selection/IdFM/
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle




def optimal_n_cluster(models_dict, display=False):
    """
    Use the slope heuristic to find the 'best' number of clusters for a model.
    Reference for the implementation:
    J.-P. Baudry, C. Maugis, and B. Michel, “Slope heuristics: overview and implementation,”
    Stat Comput, vol. 22, no. 2, pp. 455–470, Mar. 2012, doi: 10.1007/s11222-011-9236-1.

    Parameters
    ----------
    models_dict: dictionnary with
        keys = strings (model_name)
        values = couples (Log_likelihood, n_parameters)
    display (bool): if True, plots the same figures as in the included reference (fig. 2),
        to be used as a debug. Defaults to False.

    Returns
    -------
    best_model: string (one of the keys of models_dict)

    """

    linear_regressions =  []
    chosen_models_list = [] # we will obtain one string per regression
    ordered_models = list(models_dict.keys())
    ordered_models.sort(key=lambda k:-models_dict[k][1]) # we take minus the number of parameters to obtain a decreasing order
    n_pts_array = np.arange(3, len(models_dict)+1)


    LL_array =           np.array([models_dict[model_name][0] for model_name in ordered_models])
    n_parameters_array = np.array([models_dict[model_name][1] for model_name in ordered_models])

    # Step 1: compute the linear regressions
    for n_pts in n_pts_array:
        linear_reg = LinearRegression()
        linear_reg.fit(n_parameters_array[:n_pts].reshape(-1,1), LL_array[:n_pts])

        linear_regressions.append(linear_reg)
        estimated_slope = linear_reg.coef_

        chosen_slope = 2* estimated_slope
        penalized_LL = LL_array - chosen_slope * n_parameters_array

        chosen_model_index = np.argmax(penalized_LL)
        chosen_model = ordered_models[chosen_model_index]
        chosen_models_list.append(chosen_model)


    # Step 2: find the best n_clusters
    # (we could have done all steps at once, but things are clearer this way)
    constant_sequences = []
    previous_chosen_model = None
    n_successive_points = 0
    for n_pts in n_pts_array:
        chosen_model = chosen_models_list[n_pts-3]
        if chosen_model == previous_chosen_model:
            n_successive_points += 1
        else:
            if previous_chosen_model != None:
                constant_sequences.append((previous_chosen_model, n_successive_points))
            previous_chosen_model = chosen_model
            n_successive_points = 1

    constant_sequences.append((previous_chosen_model, n_successive_points))

    # remove sequences that are 'too short'
    length_threshold = np.ceil(0.15 * len(n_pts_array))
    kept_sequences = [(n_c, length) for (n_c, length) in constant_sequences if length >= length_threshold]

    if len(kept_sequences) == 0:
        print("model selection warning: optimal_n_cluster failed to find a good choice for the number of clusters. It will return the least bad")
        longest_sequence = max(constant_sequences, key=lambda c:c[0])
        best_n_clusters = longest_sequence[0]
    else:
        best_n_clusters = kept_sequences[0][0]
            # the first elements of the list were computed with the least number of points



    if display:
        plt.figure(figsize=(15, 8))
        plt.subplot(1,3,1)
        plt.title("LL vs. model size")
        plt.xlabel("$n_{param}$")
        plt.grid(True)
        plt.scatter(n_parameters_array, LL_array, c=[[0.71, 0.16, 0.38]])

        for i, lr in enumerate(linear_regressions):
            color = [1-i/(len(linear_regressions)-1), 0, i/(len(linear_regressions)-1)]
            line = lr.predict(n_parameters_array.reshape(-1,1))
            plt.plot(n_parameters_array, line, c=color, linewidth=0.2)

        plt.subplot(2,3,2)
        plt.plot(n_pts_array, [lr.coef_ for lr in linear_regressions])
        plt.xticks(n_pts_array, n_pts_array.astype(int))
        plt.gca().set_xlim(len(n_parameters_array)+1,2)
        plt.title("Slope")

        plt.subplot(2,3,5)
        n_param_chosen_model = np.array([models_dict[model_name][1] for model_name in chosen_models_list])
        plt.plot(n_pts_array,   n_param_chosen_model, "d")
        plt.xticks(n_pts_array, n_pts_array.astype(int))
        ytick_pos = np.unique(n_param_chosen_model)
        np.sort(ytick_pos)
        ytick_label = [[model_name for (model_name, (_, n_params_model)) in models_dict.items() if n_params == n_params_model][0] for n_params in ytick_pos]
        plt.yticks(ytick_pos, ytick_label)
        plt.grid(True)
        plt.title("chosen number of clusters")
        plt.gca().set_xlim(len(n_parameters_array)+1,2)
        plt.xlabel("number of points for the regression")

        plt.subplot(1,3,3)
        plt.title("penalized LL vs. model size")
        plt.xlabel("$n_{param}$")
        plt.grid(True)

        for i, lr in enumerate(linear_regressions):
            color = [1-i/(len(linear_regressions)-1), 0, i/(len(linear_regressions)-1)]

            estimated_slope = lr.coef_
            chosen_slope = 2* estimated_slope
            penalized_LL = LL_array - chosen_slope * n_parameters_array

            i_best = np.argmax(penalized_LL)
            assert (chosen_models_list[i] ==  ordered_models[i_best])
            chosen_model = chosen_models_list[i]
            plt.plot(n_parameters_array, penalized_LL, c=color, linewidth=0.2)
            plt.scatter([n_parameters_array[i_best]], [penalized_LL[i_best]], c=[color], marker='d', s=15)


    return best_n_clusters


#%%




if __name__ == "__main__":


    results = {}
    filename = lambda x:Path(f"results/n_clusters_selection/IdFM/results_{x[0]}_clusters_{x[1]}_segments.pickle")

    for nc in range(1,30):
        for ns in range(1,30):
            print(filename((nc, ns)))
            try:
                with open(filename((nc, ns)), "rb") as f:
                    this_result = pickle.load(f)
                if type(this_result) == dict:
                    results.update(this_result)
            except:
                _=5 # do nothing

    models_dict = {}
    for n_clust_n_seg in results:
        average_LL = np.mean([tup[1] for tup in results[n_clust_n_seg]])  # model, LL_train, LL_val, ...
        n_params = results[n_clust_n_seg][0][0].n_parameters  # we assume all initializations have the same number of parameters :)
        models_dict[n_clust_n_seg] = (average_LL, n_params)


    optimal_n_cluster(models_dict, display=True)

    print({(nc, ns):models_dict[(nc, ns)] for (nc, ns) in models_dict if ns == 2})


#%%   Plot one example of regression



    all_LL       = np.array([models_dict[(nc, ns)][0] for nc, ns in models_dict if (ns == 2)])
    all_n_params = np.array([models_dict[(nc, ns)][1] for nc, ns in models_dict if (ns == 2)])

    LL       = np.array([models_dict[(nc, ns)][0] for nc, ns in models_dict if ((ns == 2) and (nc >= 12))])
    n_params = np.array([models_dict[(nc, ns)][1] for nc, ns in models_dict if ((ns == 2) and (nc >= 12))])


    n_clusters_list = [nc for nc, ns in models_dict if (ns == 2)]
    n_clusters_list.sort()

    lr = LinearRegression()
    lr.fit(n_params.reshape(-1,1), LL)

    LL_predicted = lr.predict(all_n_params.reshape(-1,1))
    LL_penalized = all_LL - 2*LL_predicted + 2*lr.intercept_


    plt.figure(figsize=(7, 7))
    plt.grid(True)


    plt.scatter(all_n_params, all_LL,       c=[[0.71, 0.16, 0.38]], label="log-likelihood",    zorder=50)
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    plt.xlabel("number of clusters $K$")
    plt.ylabel("log-likelihood")

    n_weights_borders = np.array([[0], [2*np.max(all_LL)]])
    LL_borders = lr.predict(n_weights_borders)
    plt.plot(n_weights_borders, LL_borders, c=[0.65, 0.86, 0.07], label="linear regression", zorder=20)

    plt.scatter(all_n_params, LL_penalized, c=[[0.56, 0.56, 0.87]], label="penalized criterion", zorder=50)
    i_max = np.argmax(LL_penalized)
    # plt.scatter(all_n_params[i_max], LL_penalized[i_max], c=[[0.56, 0.56, 0.87]], marker="d", label="maximum of the\npenalized criterion", zorder=100, s=100)

    plt.legend()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    n_clusters_list = np.array([nc for nc, ns in models_dict if (ns == 2)] +[19])
    ticklabels = np.array([(str(k) if (k in [1,20] or k%3 ==0) else '') for k in n_clusters_list])
    select = np.array([len(t) > 0 for t in ticklabels])
    plt.xticks(all_n_params[select], ticklabels[select])

    plt.savefig("results/figures/TRB/example_regression.pdf", bbox_inches='tight')
