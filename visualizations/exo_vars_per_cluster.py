import textwrap
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib




def ind_var_per_cluster(r, ind_variables, relative=False, colors_per_cluster=None, variable_names=None, legend_answers={},
                        ylabel="number of individuals", titles={}):
    """
    Parameters
    ----------
    r: np array with shape (n_individuals, n_days, n_clusters, n_segments)
        the cluster soft-assignment
    ind_variables:  np array with shape (n_individuals,  n_ind_variables)
    relative: bool, optional
        If True, instead of showing a bar plot with a number of individuals, we show a
        relative difference betwee the proportions of each value per cluster and the
        proportions of each value in the dataset (in %). Defaults to False
    colors_per_cluster: np array with shape (n_clusters*n_segments, 3), optional
        if not provided, a new set of colors is generated at random
    variable_names: list of strings or dict, optional. A dictionary must have following structure:
        keys: "individual" and "temporal"
        values: list of strings
            the order of the strings matches the order of the columns in the variable matrices.
        If absent, the varoiable names are replaced with ind_variable_0, ind_variable_1, temp_variable_0, etc.
    legend_answers: dict with
        key = name of the variables (strings)
        value = dict with
            key = int
            value = string, the answer to the question
        Some of the survey questions call for categorical answers, which are noted using numbers (1,2,3 ..)
    ylabel: string, optional
        Defaults to "number of individuals".
    titles: dict, optional
        keys = variable names
        values = strings
        if omitted, the variable names are used as titles


    Returns
    -------
    None.

    """

    n_individuals, _, n_clusters, n_segments = r.shape
    r_clusters = np.nansum(np.nanmean(r, axis=1), axis=2)  # size: (n_individuals, n_clusters)
    clusters = np.argmax(r_clusters, axis=1)
    # n_columns = ind_variables.shape[1]  # = n_ind_variables
    if variable_names is None:
        variable_names = {"individual":[ f"ind_variable_{i}" for i in range(len(variable_names["individual"]))],
                          "temporal"  :[f"temp_variable_{i}" for i in range(len(variable_names["temporal"  ]))] }
    if type(variable_names) == dict:
        ind_var_names = variable_names["individual"]
    elif type(variable_names) == list:
        ind_var_names = variable_names
    n_ind_var = len(ind_var_names)

    # we 'update' the titles giving priority to the argument
    titles_provided = titles
    titles = {v_name:v_name for v_name in ind_var_names}
    titles.update(titles_provided)

    if colors_per_cluster is None:
        colors_per_cluster = np.random.uniform(0,1, (n_clusters, 3))
    else:
        colors_per_cluster = colors_per_cluster.reshape(n_clusters, n_segments, 3)
        colors_per_cluster = colors_per_cluster[:,0,:]  # first segment


    discrete_variable = {var_name:False for var_name in ind_var_names}
    continuous_bins = {}
    unique_values = {}  # for discrete variables only
    if relative: global_hist = {}
    for i_var, var_name in enumerate(ind_var_names):
        this_var = ind_variables[:, i_var]
        this_var = this_var[~np.isnan(this_var)]
        print(var_name[:20], np.isnan(this_var).sum(), np.isnan(this_var).mean())
        discrete_variable[var_name] = (len(np.unique(this_var)) < 10)  and np.allclose(this_var, np.round(this_var))
        # print("\n")
        # print(var_name[:20], len(np.unique(this_var)) ,  np.allclose(this_var, np.round(this_var)))
        if discrete_variable[var_name]:
            unique_values[var_name] = np.unique(this_var)
            if relative:
                values_answer = np.unique(this_var)
                values_answer.sort()
                global_hist[var_name] = np.zeros(len(values_answer))
                for i_value, value in enumerate(values_answer):
                    global_hist[var_name][i_value] = np.nanmean(this_var == value)

        else: # continuous value
            p1  = np.percentile(this_var, 0.1)
            p99 = np.percentile(this_var, 99.9)
            bin_centers = np.linspace(p1, p99, 8)
            deltabins = bin_centers[1] - bin_centers[0]
            continuous_bins[var_name] = np.concatenate([bin_centers-deltabins/2, bin_centers[-1:]+deltabins/2], axis=0)
            if relative:
                hist, _ = np.histogram(this_var, bins=continuous_bins[var_name])
                global_hist[var_name] = hist / n_individuals


    plt.figure(figsize=(20,13))
    for k in range(n_clusters):
        these_samples = (clusters == k)

        for i_var, var_name in enumerate(ind_var_names):
            plt.subplot(n_clusters, n_ind_var, n_ind_var*k + i_var + 1)

            this_var = ind_variables[these_samples, i_var] # shape: (n_individuals_k,)
            this_var = this_var[~np.isnan(this_var)]
            # print(k, var_name[:20])
            if discrete_variable[var_name]:
                values_answer = unique_values[var_name]
                values_answer.sort()
                for i_value, value in enumerate(values_answer):
                    n_ind_value = np.nanmean(this_var == value)
                    if relative:
                        bar_height = 100*(n_ind_value/this_var.shape[0] - global_hist[var_name][i_value]) #/ (global_hist[var_name][i_value] + 0.001)
                    else:
                        bar_height = n_ind_value

                    plt.bar(i_value, bar_height, width=1,
                        edgecolor=colors_per_cluster[k,:], facecolor=list(colors_per_cluster[k,:])+[0.5]) # 0.5 = alpha

                if k == n_clusters -1: # last line
                    if var_name in legend_answers:
                        ticklabels = [legend_answers[var_name][v] for v in values_answer]
                        tick_loc = np.arange(len(ticklabels))
                    else:
                        ticklabels = [str(v) for v in values_answer]
                        tick_loc = values_answer

                    if np.any([len(l) > 10 for l in ticklabels]):
                        plt.xticks(tick_loc, ticklabels, rotation=-45, ha='left')
                    else:
                        plt.xticks(tick_loc, ticklabels)

            else: # continuous value
                if relative:
                    hist, _ = np.histogram(this_var, bins=continuous_bins[var_name])
                    proportion_hist = hist / this_var.shape[0]
                    relative_hist = 100*(proportion_hist - global_hist[var_name]) #/ (global_hist[var_name] + 0.001)
                    bar_width = continuous_bins[var_name][1] - continuous_bins[var_name][0]
                    bin_centers = (continuous_bins[var_name][1:] + continuous_bins[var_name][:-1])/2
                    plt.bar(bin_centers, relative_hist, width=bar_width,
                        edgecolor=colors_per_cluster[k,:], facecolor=list(colors_per_cluster[k,:])+[0.5]) # 0.5 = alpha

                else:
                    plt.hist(this_var, bins=continuous_bins[var_name],
                        edgecolor=colors_per_cluster[k,:], facecolor=list(colors_per_cluster[k,:])+[0.5])  # 0.5 = alpha
                if k == n_clusters -1: # try a legend
                    bin_centers = (continuous_bins[var_name][1:] + continuous_bins[var_name][:-1])/2
                    bin_centers = bin_centers[::2]
                    bin_centers = np.round(bin_centers).astype(int)
                    plt.xticks(bin_centers, bin_centers)

            if i_var == 0:
                plt.ylabel(ylabel)
            if k==0:
                title_w_newlines = '\n'.join(textwrap.wrap(titles[var_name], 30)) # thanks to jfs for this answer
                    # https://stackoverflow.com/questions/2657693/insert-a-newline-character-every-64-characters-using-python
                plt.title(title_w_newlines)
            if k != n_clusters -1:
                plt.xticks([], [])



legend_answers = {
    'electric_heating':{
        0:'False',
        1:'True'},

    'Question 310: What is the employment status of the chief income earner in your household, is he/she':{
        0: "missing",
    	1: "An employee",
    	2: "Self-employed (with employees)",
    	3: "Self-employed (with no employees)",
    	4: "Unemployed (actively seeking work)",
    	5: "Unemployed (not actively seeking work)",
    	6: "Retired",
    	7: "Carer: Looking after relative family"},

    'Question 401: SOCIAL CLASS Interviewer, Respondent said that occupation of chief income earner was.... <CLASS> Please code':{
        0: "missing    ",
    	1: "AB",
    	2: "C1",
    	3: "C2",
    	4: "DE",
    	5: "F",
    	6: "Refused"},


    'Question 430: And how many of these are typically in the house during the day (for example for 5-6 hours during the day)':{
        0: '0',
    	1: "1",
    	2: "2",
    	3: "3",
    	4: "4",
    	5: "5",
    	6: "6",
        7: "7",
        8: "   None"},
    }



    # 'Question 43111: How many people under 15 years of age live in your home?':{
    #     0: "0",
    # 	1: "1",
    # 	2: "2",
    # 	3: "3",
    # 	4: "4",
    # 	5: "5",
    # 	6: "6",
    # 	7: "7+" },
# QUESTION 470 	MULTIPLE
# Which of the following best describes how you heat your home?
# 	1		Electricity (electric central heating storage heating)
# 	2		Electricity (plug in heaters)
# 	3		Gas
# 	4		Oil
# 	5		Solid fuel
# 	6		Renewable (e.g. solar)
# 	7		Other

# QUESTION 4701 	MULTIPLE
# Which of the following best describes how you heat water in your home?
# 	1		Central heating system
# 	2		Electric (immersion)
# 	3		Electric (instantaneous heater)
# 	4		Gas
# 	5		Oil
# 	6		Solid fuel boiler
# 	7		Renewable (e.g. solar)
# 	8		Other




titles = {
    'electric_heating': 'Use of electrical heating',
    'n_people_per_ind': 'Number of people per household', # (children count for half an adult)',
    "n_children":  "Number of children",
    'Question 310: What is the employment status of the chief income earner in your household, is he/she':
        "Employment of the 'chief income earner'",
    'Question 401: SOCIAL CLASS Interviewer, Respondent said that occupation of chief income earner was.... <CLASS> Please code':
        "Social class",
    'Question 430: And how many of these are typically in the house during the day (for example for 5-6 hours during the day)':
        "Number of adults staying at home during the day",
    'Question 43111: How many people under 15 years of age live in your home?':
        "Number of childrens ",
    }



replacing_question = 'Question 430: And how many of these are typically in the house during the day (for example for 5-6 hours during the day)'
ind_replacing = variable_names["individual"].index(replacing_question)
ind_variables_train[ind_variables_train[:,ind_replacing] == 8] = 0


variables_to_plot = list(legend_answers.keys()) + ["n_people_per_ind", 'electric_heating', "n_children"]
plotted_ind_var, _, sel_var_names = select_variables(variables_to_plot, ind_variables_train, temp_variables_train, variable_names)
ind_var_per_cluster(np.exp(log_r_train), plotted_ind_var, colors_per_cluster=base_colors_per_cluster[:model.n_clusters,:],
                    variable_names=sel_var_names, legend_answers=legend_answers, ylabel="number of households", titles=titles)


ind_var_per_cluster(np.exp(log_r_train), plotted_ind_var, relative=True, colors_per_cluster=base_colors_per_cluster[:model.n_clusters,:],
                    variable_names=sel_var_names, legend_answers=legend_answers, ylabel="share of households (%)", titles=titles)

# QUESTION 410
# What best describes the people you live with?
# READ OUT

# 	1		I live alone
# 	2		All people in my home are over 15 years of age
# 	3		Both adults and children under 15 years of age live in my home
# TODO n_children




