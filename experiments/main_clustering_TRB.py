import datetime
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from unidecode import unidecode  # remove diacritics

from utils.nan_operators import nan_logsumexp
from utils.logistic_regression import softmax
from utils.data_generation import generate_data, generate_inconsistent_data
from preprocessing.IdFM.load_data import load_data
from visualizations.clustering_results import plot_cluster_ind_day
from visualizations.location_data_IdFM import plot_clusters_location
from visualizations.significant_difference import get_significant_differences

from models.main_model import GMM_segmentation




np.random.seed(0)


chosen_variables =  ["strike", "strike_2019", "lockdown_different", "school_holiday", "national_holiday",
                      "year_splines_d2_n21", "day_of_week", "time"]




#%%



data, variable_names, id_names, IdFM_origin = load_data("rail", ticket_types="merge", start_date=datetime.date(2017,1,1), sampling=None,
                                                chosen_variables=chosen_variables, normalisation="individual",
                                                remove_mean="none", scale='log10', verbose=True)


(Y, ind_array, day_array, variables) = data[0]



n_ind_variables  = variables[0].shape[2]
n_temp_variables = variables[1].shape[2]
n_mixed_variables = 0


n_individuals = Y.shape[0]
n_days = Y.shape[1]





#%%



n_clusters = 9
n_segments = 2

model = GMM_segmentation(dim=Y.shape[2], n_clusters=n_clusters, n_segments=n_segments,
                          n_variables=(n_ind_variables, n_temp_variables, 0),
                          covariance_type="diag", min_slope=4.4 * n_days/90)

LL_list = model.EM(Y,  variables,
                    init_with_CEM=False, init_method=('parameters', ("random", "uniform")),
                    CEM=False, print_every=1)





#%%
# Reorder the clusters by the effect of weekday vs weekend

weekday_list = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday',]
weekday_inds = np.array([variable_names["temporal"].index(day_name) for day_name in weekday_list])
weekend_inds = np.array([variable_names["temporal"].index(day_name) for day_name in ['is_Saturday', 'is_Sunday']])


effect_weekday = model.beta[weekday_inds,0,:,0].mean(axis=0)  # shape: (n_clusters)
effect_weekend = model.beta[weekend_inds,0,:,0].mean(axis=0)  # shape: (n_clusters)
total_effect = effect_weekday - effect_weekend
new_order = np.argsort(-total_effect)  # order ascending needs minus sign

model.pi = model.pi[new_order]
model.m  = model.m[ :,new_order,:]
model.alpha = model.alpha[:,:,new_order,:]
model.beta  = model.beta[ :,:,new_order,:]
model.gamma = model.gamma[:,:,new_order,:]
model.sigma = model.sigma[:,new_order,:]
model.u = model.u[new_order,:]
model.v = model.v[new_order,:]


colors = np.zeros((model.n_clusters*model.n_segments, 3))
colors_per_cluster = np.zeros((model.n_clusters, 3))  # colors[::model.n_segments,:]

for i in range(model.n_clusters):
    color_to_use = i if i < 7 else i+1 # skip grey
    colors_per_cluster[i,:] = colormaps["tab10"](0.05 + 0.1*color_to_use)[:3]

colors[ ::model.n_segments,:] = (colors_per_cluster*1 + 1*1) /2
colors[1::model.n_segments,:] = (colors_per_cluster*1 + 0*1) /2


if model.n_segments == 3:
    colors[2::model.n_segments,:] = colors_per_cluster



#%%

# =============================================================================
#                 Cluster and segment assignment image
# =============================================================================


days_legend = {i*365:datetime.date(year=IdFM_origin.year + i, month=1, day=1) for i in range(5+1) }


displayed_periods = {(datetime.date(day=5,  month=12, year=2019), datetime.date(day=13, month=1,  year=2020)):("r", "2019 strike"),
                     (datetime.date(day=17, month=3,  year=2020), datetime.date(day=11, month=5,  year=2020)):([0., 1., 0.],   "Lockdown 1"),
                     (datetime.date(day=30, month=10, year=2020), datetime.date(day=15, month=12, year=2020)):([0.3, 1., 0.3], "Lockdown 2"),
                     (datetime.date(day=3,  month=4,  year=2021), datetime.date(day=3,  month=5,  year=2021)):([0.6, 1., 0.6], "Lockdown 3")}
        # keys = couples of dates, values = (color, name)

displayed_periods = {tuple([(k[i]-IdFM_origin).days for i in [0,1]]):v for k,v in displayed_periods.items()}
        # keys = coupkes of ints (number of days), values = (color, name)


contributions = (model.alpha, model.beta, model.gamma)
log_r, _ = model.E_step(Y, variables)
r = np.exp(log_r)

plt.figure(figsize=(7, 5))
plot_cluster_ind_day(np.exp(log_r), colors_per_cluster=colors,
                              ind_ordering="majority_cluster", new_figure=False,
                              days_legend=days_legend, displayed_periods=displayed_periods,
                              ylabel="stations")
plt.tight_layout()
plt.gca().get_xticklabels(minor=True)[0].set_color("red")
plt.gca().get_xticklabels(minor=True)[1].set_color([0., 0.7, 0.])
plt.gca().get_xticklabels(minor=True)[2].set_color([0., 0.7, 0.])
plt.gca().get_xticklabels(minor=True)[3].set_color([0., 0.7, 0.])


plt.xlabel(" "*8 + "segment 1" + " "*50 + "segment 2"  +"\n"+"date")


plt.savefig("results/figures/TRB/IdFM_cluster_segments.pdf", bbox_inches='tight', dpi=900)










#%%


# =============================================================================
#                      Residuals
# =============================================================================


log_r_ind_day_clust = nan_logsumexp(log_r, axis=3, all_nan_return=np.nan)  # shape = (n_individuals, n_days, n_clusters)

log_r_ind_clust = np.nansum(log_r_ind_day_clust, axis=1)  # shape = (n_individuals, n_clusters)
log_r_ind_clust = log_r_ind_clust.copy()
log_r_ind_clust[np.isnan(log_r_ind_clust)] = -np.inf

global_residuals = np.zeros((n_individuals, n_days, model.dim))
chosen_cluster = np.argmax(log_r_ind_clust, axis=1)

per_cluster_residuals = model.get_residuals(Y, variables)



for k in range(model.n_clusters):
    these_inds = (chosen_cluster==k)


    this_Y = Y[these_inds,...]

    log_r_segment = log_r[these_inds,:,k,:]
    log_r_segment = log_r_segment.copy()


    log_r_segment[np.isnan(log_r_segment)] = -np.inf # np.argmax considers nan is +inf
    chosen_s = np.argmax(log_r_segment, axis=2)

    for s in range(model.n_segments):
        these_t_i = (chosen_s == s)

        mask = np.zeros((these_inds.sum(), n_days))
        mask[these_t_i] = 1

        global_residuals[these_inds,:,:] += mask[:,:,np.newaxis] * per_cluster_residuals[these_inds,:,:,k,s]


all_days_datetimes = [IdFM_origin + datetime.timedelta(days=d) for d in range(n_days)]
mean = np.nanmean(global_residuals[:,:,0], axis=0)
std  =  np.nanstd(global_residuals[:,:,0], axis=0)


plt.figure(figsize=(10, 10))
plt.plot(all_days_datetimes, mean, c=[1., 0., 0.])
plt.fill_between(all_days_datetimes, mean-std, mean+std, facecolor=[1., 0., 0., 0.3])
plt.gca().set_ylim([-1.51, 1.01])
plt.title("mean +/- std of the per-sample residuals (computed using binarized responsibilities)")
plt.grid(True)


order = np.argsort(-np.abs(mean))  # sort ascending
print("largest residuals:")
n_lowest = 20
for i in range(n_lowest):
    print("- ", all_days_datetimes[order[i]], f": {mean[order[i]]:.5f} +/- {std[order[i]]:.2f}")






#%%

# =============================================================================
#             Plotting a few select staitons
# =============================================================================

station_names = ["GARE DE LYON", "LA DEFENSE-GRANDE ARCHE", "LE GUICHET" ]


def select_by_name(selected_station_names, ind_array):
    """
    Parameters
    ----------
    selected_station_names: either a string (single name) or a list of strings
    ind_array: np.array()


    Returns
    -------
    select_array: np array of bools with shape: (n_individuals,)
        Should be used directly as Y[select_array,:,:]
    """

    if type(selected_station_names) == str:
        selected_station_names = [selected_station_names]
    # now station_names is a list of strings
    selected_station_names_lower = [unidecode(s.lower()) for s in selected_station_names]

    selected_IDs = []
    for id_station, name_list in id_names.items():
        for name in name_list:  # some stations may have changed names or denomination
            for selected_station_name in selected_station_names_lower:
                if selected_station_name in name.lower():
                    selected_IDs.append(id_station)  # we may have duplicates, we do not care

    return np.isin(ind_array, selected_IDs)



for station_name in station_names:
    ind_filter = select_by_name([station_name], ind_array)
    assert ind_filter.sum() == 1
    i_ind = np.argmax(ind_filter)  # stop at first True
    k = chosen_cluster[i_ind]

    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot(all_days_datetimes, Y.reshape(n_individuals, -1)[i_ind,:], c='r', label='data')
    reconstruction =  Y.reshape(n_individuals, -1)[i_ind,:] - global_residuals.reshape(n_individuals, -1)[i_ind,:]
    plt.plot(all_days_datetimes, reconstruction, c='b', label='model reconstruction', alpha=0.5)



    plt.legend()
    plt.grid(True)
    plt.title(f"station '{station_name}'")


    plt.subplot(2,1,2)
    plt.title("Residual")
    plt.plot(all_days_datetimes, global_residuals.reshape(n_individuals, -1)[i_ind,:])
    plt.grid(True)




#%%

old_default_fontsize = matplotlib.rcParams['font.size']
matplotlib.rcParams.update({'font.size': 13})

station_names = ["LA DEFENSE-GRANDE ARCHE", "GARE DE LYON", "AEROPORT CHARLES DE GAULLE 2", "LE GUICHET"]

station_colors = {"LA DEFENSE-GRANDE ARCHE":      [1., 0., 0.],
                   "GARE DE LYON":                [0., 1., 0.],
                  "AEROPORT CHARLES DE GAULLE 2": [0., 0., 1.],
                 "LE GUICHET":                    [0.7, 0.7, 0.],
                  }


station_zorder = {"LA DEFENSE-GRANDE ARCHE":      10,
                   "GARE DE LYON":                20,
                  "AEROPORT CHARLES DE GAULLE 2": 30,
                  "LE GUICHET":                    0,
                  }


replacement_names = {"LA DEFENSE-GRANDE ARCHE": "LA DEFENSE-GRANDE ARCHE",
                   "GARE DE LYON": "GARE DE LYON",
                   "AEROPORT CHARLES DE GAULLE 2": "CHARLES DE GAULLE AIRPORT 2",
                   "LE GUICHET":"LE GUICHET"}

plt.figure(figsize=(12, 4))
main_ax = plt.gca()
sub_ax = plt.axes([0.2,0.2,0.2,0.2])
period_zoom = slice(253, 253+15)



for station_name in station_names:

    ind_filter = select_by_name([station_name], ind_array)
    assert ind_filter.sum() == 1
    i_ind = np.argmax(ind_filter)
    k = chosen_cluster[i_ind]

    main_ax.plot(all_days_datetimes, Y.reshape(n_individuals, -1)[i_ind,:],
                 label=replacement_names[station_name], linewidth=1, c=station_colors[station_name],
                 zorder=station_zorder[station_name])
    sub_ax.plot(all_days_datetimes[period_zoom], Y.reshape(n_individuals, -1)[i_ind,period_zoom],
                label=station_name, linewidth=1, c=station_colors[station_name])


main_ax.legend(loc='lower right')
main_ax.grid(True)
main_ax.set_xlabel("date")
main_ax.set_ylabel(r"normalized log$_{10}$ of the" +"\n number of entries", rotation=90)

sub_ax.grid(True)
sub_ax.set_yticks([],[])
sub_ax.set_xticks(all_days_datetimes[period_zoom],
                  [(f'{date.day:02d}-{date.month:02d}'  if date.weekday() ==0 else '') for date in all_days_datetimes[period_zoom]])




plt.savefig("results/figures/TRB/IdFM_data_example.pdf", bbox_inches='tight')

matplotlib.rcParams.update({'font.size': old_default_fontsize})




 #%%

# =============================================================================
#            Empirical effect of the lockdown
# =============================================================================



Y_non_normalized = load_data("rail", ticket_types="merge", start_date=IdFM_origin, sampling=None,
                                                chosen_variables=[], normalisation="none",
                                                remove_mean="none", scale='linear')[0][0][0]
total_per_day = np.nansum(Y_non_normalized, axis=(0,2))

plt.figure(figsize=(6, 6))

any_lockdown = np.zeros(n_days, dtype=bool) # True during lockdowns, False otherwise
        #initialized with False everywhere


start_baseline = (datetime.date(2018,12,1) - IdFM_origin).days
end_baseline   = (datetime.date(2019,12,1) - IdFM_origin).days

baseline = np.mean(total_per_day[start_baseline:end_baseline])

dates_start = {}
dates_end   = {}
for i_l in [1,2,3]:
    index_lockdown = variable_names["temporal"].index(f"lockdown_{i_l}")
    coef_per_segment = model.beta[index_lockdown,0,k,:]   # shape: (n_segments,)
    is_lockdown = (variables[1][0,:,index_lockdown]>0)  # shape: (n_days,)
    any_lockdown += is_lockdown
    avg_value = np.nanmean(total_per_day[is_lockdown])
    variation = 100 * (avg_value - baseline) / baseline

    plt.bar(x=i_l, width=0.6, bottom=0, height=variation,
            facecolor=[0.4, 0.4, 0.4])

    day_start = int(np.where(np.diff(is_lockdown.astype(int)) > 0)[0][0])
    dates_start[i_l] = IdFM_origin + datetime.timedelta(days=day_start)
    day_end = int(np.where(np.diff(is_lockdown.astype(int)) < 0)[0][0])
    dates_end[i_l] = IdFM_origin + datetime.timedelta(days=day_end)


after_pandemic = np.array([d >= datetime.date(2020,5,11) for d in all_days_datetimes])
non_lockdown_pandemic = (after_pandemic & (~any_lockdown))

avg_value = np.nanmean(total_per_day[non_lockdown_pandemic])
variation = 100 * (avg_value - baseline) / baseline
plt.bar(x=4, width=0.6, bottom=0, height=variation,
        facecolor=[0.4, 0.4, 0.4])




ticklabels = [f"Lockdown {i_l} \n({dates_start[i_l]}, \n{dates_end[i_l]})" for i_l in [1,2,3]] + ["Rest of the\npandemic"]
plt.ylabel("Variation (%)")
plt.xticks(np.arange(1,5), ticklabels)

plt.savefig("results/figures/TRB/IdFM_lockdowns.pdf", bbox_inches='tight')





#%%
















#%%


# =============================================================================
#             Comparing real clusters with simulated ones
# =============================================================================


mean_ind_var = variables[0] * 0 + np.nanmean(variables[0], axis=0, keepdims=True)  # shape: (n_individuals, 1, n_ind_var)

plt.figure(figsize=(40, 20))
plt.subplot(2,1,1)
main_ax = plt.gca()
sub_ax = plt.axes([0.2,0.6,0.3,0.1])



for k in range(model.n_clusters):
    simulated_Y, _ = generate_data(np.ones(1), model.u[k:k+1,:], model.v[k:k+1,:], model.m[:,k:k+1,:],
                             (model.alpha[:,:,k:k+1,:], model.beta[:,:,k:k+1,:], model.gamma[:,:,k:k+1,:]),
                             model.sigma[:,k:k+1,:], (mean_ind_var, variables[1][:,:,:], variables[2]),
                             covariance_type="infer")


    mean = np.nanmean(simulated_Y.reshape(n_individuals, -1), axis=0)  # shape: (n_days,)
    std  =  np.nanstd(simulated_Y.reshape(n_individuals, -1), axis=0)

    main_ax.plot(all_days_datetimes, mean,  "o", markersize=5,
             c=colors_per_cluster[k,:], label=f"cluster {k+1}")
    main_ax.fill_between(all_days_datetimes, mean-std, mean+std, facecolor=colors_per_cluster[k,:], alpha=0.3)

    sub_ax.plot(all_days_datetimes[period_zoom], mean[period_zoom], c=colors_per_cluster[k,:])
    sub_ax.fill_between(all_days_datetimes[period_zoom], (mean-std)[period_zoom], (mean+std)[period_zoom],
                        facecolor=colors_per_cluster[k,:], alpha=0.3)


main_ax.legend(fontsize=20)
main_ax.set_ylabel("simulated distribution", fontsize=20)
main_ax.set_ylim(-3., 0.85)
main_ax.grid(True)

sub_ax.grid(True)
sub_ax.set_yticks([],[])
sub_ax.set_xticks(all_days_datetimes[period_zoom],
                  [(f'{date.day:02d}-{date.month:02d}') for date in all_days_datetimes[period_zoom]], rotation=-25)




plt.subplot(2,1,2)
main_ax = plt.gca()
sub_ax = plt.axes([0.2,0.2,0.3,0.1])


chosen_cluster = np.argmax(log_r_ind_clust, axis=1) # shgape: (n_individuals,)
for k in range(model.n_clusters):
    this_k_Y = Y[(chosen_cluster == k),:,:]
    this_n_ind = (chosen_cluster == k).sum()
    mean = np.nanmean(this_k_Y.reshape(this_n_ind, -1), axis=0)
    std  =  np.nanstd(this_k_Y.reshape(this_n_ind, -1), axis=0)

    main_ax.plot(all_days_datetimes, mean, "o", markersize=5,
             c=colors_per_cluster[k,:], label=f"cluster {k+1}")
    main_ax.fill_between(all_days_datetimes, mean-std, mean+std, facecolor=colors_per_cluster[k,:], alpha=0.3)

    sub_ax.plot(all_days_datetimes[period_zoom], mean[period_zoom], c=colors_per_cluster[k,:])
    sub_ax.fill_between(all_days_datetimes[period_zoom], (mean-std)[period_zoom], (mean+std)[period_zoom],
                        facecolor=colors_per_cluster[k,:], alpha=0.3)



main_ax.legend(fontsize=20)
main_ax.set_ylabel("empirical distribution", fontsize=20)
main_ax.set_ylim(-3., 0.85)
main_ax.grid(True)

sub_ax.grid(True)
sub_ax.set_yticks([],[])
sub_ax.set_xticks(all_days_datetimes[period_zoom],
                  [(f'{date.day:02d}-{date.month:02d}') for date in all_days_datetimes[period_zoom]], rotation=-25)





#%%


plt.figure(figsize=(30, 20))
subplot_list = []
ymin, ymax = np.inf, -np.inf

xmin = (datetime.date(2020,3,1) - IdFM_origin).days
xmax = (datetime.date(2020,8,1) - IdFM_origin).days



generated_data = generate_inconsistent_data(r, model.m, contributions, model.sigma*0, variables)
        #we set the std to zero and will plot the variance iuesigma separaterlye


for k in range(model.n_clusters):
    mean_exp = np.nanmean(Y[(chosen_cluster == k),xmin:xmax,0], axis=0)   # shape: (xmax-xmin,)
    std_exp  =  np.nanstd(Y[(chosen_cluster == k),xmin:xmax,0], axis=0)   # shape: (xmax-xmin,)

    mean_model = np.nanmean(generated_data[(chosen_cluster == k),xmin:xmax,0], axis=0)    # shape: (xmax-xmin,)
    std_model  = np.nansum(r[(chosen_cluster == k),xmin:xmax,k,:] @ np.sqrt(model.sigma[0,k,:]), axis=0) /\
                            np.nansum(r[(chosen_cluster == k),xmin:xmax,k,:], axis=(0,2)) # shape: (xmax-xmin,)



    this_subplot = plt.subplot(np.ceil(model.n_clusters/2).astype(int),2,k+1)
    subplot_list.append(this_subplot)

    plt.plot(all_days_datetimes[xmin:xmax], mean_exp, "o", markersize=5,
              c="k", zorder=-3, label="empirical mean") #colors_per_cluster[k,:])
    plt.fill_between(all_days_datetimes[xmin:xmax], mean_exp-std_exp, mean_exp+std_exp,
                      facecolor='k', alpha=0.3, step="mid", label="empirical standard deviation")

    plt.plot(all_days_datetimes[xmin:xmax], mean_model, markersize=5,
              c=colors_per_cluster[k,:], label="model mean")
    plt.errorbar(all_days_datetimes[xmin:xmax], mean_model, yerr=std_model,
                      c=colors_per_cluster[k,:], label="model standard deviation")

    plt.grid(True)
    this_ymin, this_ymax = plt.gca().get_ylim()
    ymin = this_ymin if this_ymin < ymin else ymin
    ymax = this_ymax if this_ymax > ymax else ymax



for spt in subplot_list:
    spt.set_ylim(ymin, ymax)
    # spt.set_ylim(-0.8, 0.5)


#%%

# =============================================================================
#             Zoom on cluster 3
# =============================================================================


plt.figure(figsize=(20,10))

k = 2  # beware Python indexing
for i in range(n_individuals):
    if chosen_cluster[i] == k:
        plt.plot(all_days_datetimes, Y[i,:,0], c=colors_per_cluster[k,:],
                 lw=0.2)


plt.grid(True)
plt.xlabel("date")
plt.ylabel(" Normalized $log_{10}$ entries")
plt.title("Time series of the stations assigned to cluster 3")











#%%


# =============================================================================
#             Zoom on cluster 6
# =============================================================================

periods = [{'xmin':(datetime.date(2020,3,1) - IdFM_origin).days,
            'xmax':(datetime.date(2020,10,1) - IdFM_origin).days,
            'event_name':"Lockdown 1",
            'event_start':datetime.date(2020, 3, 17),
            'event_end'  :datetime.date(2020, 5, 11),},

            # {'xmin':(datetime.date(2020,9,1) - IdFM_origin).days,
            #  'xmax':(datetime.date(2021,2,1) - IdFM_origin).days,
            #  'event_name':"Lockdown 2",
            #  'event_start':datetime.date(2020, 10, 30),
            #  'event_end'  :datetime.date(2020, 12, 15),},


            # {'xmin':(datetime.date(2021,3,1) - IdFM_origin).days,
            # 'xmax':(datetime.date(2021,10,1) - IdFM_origin).days,
            # 'event_name':"Lockdown 3",
            # 'event_start':datetime.date(2021, 4, 3),
            # 'event_end'  :datetime.date(2021, 5, 3),}
        ]


n_periods = len(periods)

plt.figure(figsize=(n_periods*12, 8))
subplot_list = []
ymin, ymax = np.inf, -np.inf

generated_data = generate_inconsistent_data(r, model.m, contributions, model.sigma*0, variables)
        #we set the std to zero and will plot the variance iuesigma separaterlye

for i_period, period_dict in enumerate(periods):
    xmin, xmax = period_dict['xmin'], period_dict['xmax']

    for i_k, k in enumerate([4,5]):
        mean_exp = np.nanmean(Y[(chosen_cluster == k),xmin:xmax,0], axis=0)   # shape: (xmax-xmin,)
        std_exp  =  np.nanstd(Y[(chosen_cluster == k),xmin:xmax,0], axis=0)   # shape: (xmax-xmin,)

        mean_model = np.nanmean(generated_data[(chosen_cluster == k),xmin:xmax,0], axis=0)    # shape: (xmax-xmin,)
        std_model  = np.nansum(r[(chosen_cluster == k),xmin:xmax,k,:] @ np.sqrt(model.sigma[0,k,:]), axis=0) /\
                                np.nansum(r[(chosen_cluster == k),xmin:xmax,k,:], axis=(0,2)) # shape: (xmax-xmin,)

        this_subplot = plt.subplot(2,n_periods, i_period + n_periods*i_k +1)
        subplot_list.append(this_subplot)

        plt.plot(all_days_datetimes[xmin:xmax], mean_exp, "o", markersize=5,
                  c="k", zorder=0, label="empirical mean") #colors_per_cluster[k,:])
        plt.fill_between(all_days_datetimes[xmin:xmax], mean_exp-std_exp, mean_exp+std_exp,
                          facecolor=[0.7, 0.7, 0.7], step="mid", label="empirical standard deviation",
                          zorder=-50)

        plt.plot(all_days_datetimes[xmin:xmax], mean_model, markersize=5,
                  c=colors_per_cluster[k,:], zorder=+50)
        plt.errorbar(all_days_datetimes[xmin:xmax], mean_model, yerr=std_model,
                          c=colors_per_cluster[k,:], label="model mean and \nstandard deviation",
                          elinewidth=0.8, zorder=+100)

        plt.grid(True)
        this_ymin, this_ymax = plt.gca().get_ylim()
        ymin = this_ymin if this_ymin < ymin else ymin
        ymax = this_ymax if this_ymax > ymax else ymax

        plt.ylabel(r"normalized $\log_{10}$ entries")

        # y_lck = -2.9  # the position of the indication for the lockdown
        # plt.plot([datetime.date(day=17, month=3,  year=2020), datetime.date(day=11, month=5,  year=2020)],
        #          [y_lck, y_lck], c=[0., 1., 0.])
        # plt.text(datetime.date(day=15, month=5,  year=2020), y_lck, 'Lockdown 1',
        #          c=[0., 1., 0.], va='center')

        plt.fill_betweenx([-10, 10],
                [period_dict["event_start"]], [period_dict["event_end"]],
                facecolor=[0.85, 1, 0.85], zorder=-100)
        plt.text(period_dict["event_start"] + datetime.timedelta(days=5),
                  -2.9, period_dict["event_name"], c=[0., 0.5, 0.], va='center')


        plt.legend(title=f'Cluster {k+1}', loc="lower right")

    plt.xlabel("date") # label on the last plot only

for spt in subplot_list:
    spt.set_ylim(ymin, ymax)


plt.savefig("results/figures/TRB/IdFM_empirical_mean_clusters.pdf", bbox_inches='tight')







#%%

plt.figure(figsize=(5,4))
for k in range(model.n_clusters):
    delta_x = (k / model.n_clusters) * 0.6 - 0.3
    for s in range(model.n_segments):
        label= f" cluster {k+1}" if s==0 else None
        plt.plot([delta_x+s+1, delta_x+s+1], [0,model.sigma[0,k,s]],
                 c=colors_per_cluster[k,:], label=label)

plt.xticks(1+np.arange(model.n_segments), 1+np.arange(model.n_segments).astype(int))
plt.xlabel("segment")

plt.legend(bbox_to_anchor=(1.04, 0), borderaxespad=0, loc='lower left')
plt.grid(axis='y')
plt.tight_layout()

plt.gca().set_ylim(0, plt.gca().get_ylim()[1])

plt.savefig("results/figures/TRB/IdFM_variances.pdf", bbox_inches='tight')




#%%



plt.figure(figsize=(model.n_segments*5,4))


unexplained_variance = model.sigma[0,:,:] # shape: (n_clusters, n_segents)
prop_explained = model.linear_regression_variance_history[-1]
total_variance = unexplained_variance / (1-prop_explained)
explained_variance = total_variance * prop_explained

subplot_list = []


for s in range(model.n_segments):
    subplot_list.append(plt.subplot(1, model.n_segments, s+1))
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Segment {s+1}")


    for k in range(model.n_clusters):
        plt.scatter([explained_variance[k,s]], [unexplained_variance[k,s]],
                    c=[colors_per_cluster[k,:]], label= f" cluster {k+1}" )

    plt.xlabel('Explained variance')
    if s == 0: plt.ylabel('Unexplained variance')


plt.legend(bbox_to_anchor=(1.04, 0), borderaxespad=0, loc='lower left')

# same scale for the first two subplots:
xmin = min([spt.get_xlim()[0] for spt in subplot_list])
xmax = max([spt.get_xlim()[1] for spt in subplot_list])

ymin = min([spt.get_ylim()[0] for spt in subplot_list])
ymax = max([spt.get_ylim()[1] for spt in subplot_list])

# plotting the iso-share-variance
n_isolines = 10
min_p = np.floor(np.min(prop_explained) * 10) / 10
max_p = np.floor( np.max(prop_explained) * 10) / 10

delta_p = max_p - min_p
p_list = np.linspace(min_p, max_p, n_isolines)

p_list = 1 - 10**np.linspace(0, -2, 11)[1:]
for p in p_list:
    x_array = np.linspace(xmin, xmax, 1000)
    y_array = ((1-p)/p) * x_array

    for spt in subplot_list:
        spt.plot(x_array, y_array, c=matplotlib.colormaps['viridis'](p), zorder=-20)


for spt in subplot_list:
    spt.set_ylim(ymin, ymax)
    spt.set_xlim(xmin, xmax)
plt.tight_layout()



plt.savefig("results/figures/TRB/IdFM_explained_variance.pdf", bbox_inches='tight')












#%%


# =============================================================================
#             Spline plot
# =============================================================================

old_colors = colors.copy()
colors[ ::model.n_segments,:] = (colors_per_cluster*4 + 1*1) /5
colors[1::model.n_segments,:] = (colors_per_cluster*4 + 0*1) /5



fig = plt.figure(figsize=(12.5,5))

selected_indices = [variable_names['temporal'].index("time")]
selected_indices += [i for i in range(n_temp_variables) if "spline" in variable_names['temporal'][i]]
selected = np.isin(np.arange(n_temp_variables), selected_indices)



for k in range(model.n_clusters):

    time = np.linspace(0, 1, n_days).reshape(-1,1)
    probas = softmax( time * model.u[k,:].reshape(1,-1) + model.v[k,:].reshape(1,-1), axis=1)  # shape: (n_days, n_segments)


    for s in range(model.n_segments):
        is_selected = (np.argmax(probas, axis=1) == s).astype(float)
        start = np.argmax(np.diff(is_selected) > 0)  # argmax stops at the first True
            # if no value is True (max == False), argmax will return zero, which is precisely what we want
        end   = np.argmax(np.diff(is_selected) < 0)
        if end == 0: end = n_days  # we need to deal with the edge case this time

        mu_selected = model.m[0,k,s] +\
                    (variables[1][:,:,selected]  @ model.beta[selected,0,k,s])[0,:]

        c = colors[k*model.n_segments+s,:]
        plt.plot(all_days_datetimes[   :start], mu_selected[   :start], c=colors[k*model.n_segments+s,:], lw=0.6, alpha=0.5)
        plt.plot(all_days_datetimes[end:     ], mu_selected[end:     ], c=colors[k*model.n_segments+s,:], lw=0.6, alpha=0.5)
        plt.plot(all_days_datetimes[start:end], mu_selected[start:end], c=colors[ k*model.n_segments+s,:],
                 label=f"cluster {k+1}, segment {s+1}")

plt.xlabel("date")
plt.ylabel(r"normalized $\log_{10}$ entries")

# plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
plt.legend(bbox_to_anchor=(0.5, 0.135), loc="upper center",
           bbox_transform=fig.transFigure, ncol=5)
plt.tight_layout()


plt.gca().set_ylim(-1., 0.5)
plt.grid(True)
plt.savefig("results/figures/TRB/reconstruction_splines.pdf", bbox_inches='tight')



colors  = old_colors.copy()












#%%


# =============================================================================
#                    Physical location of stations
# =============================================================================

dist = plot_clusters_location(log_r=log_r, id_array=ind_array, display_legends=False,
                       color_per_cluster=colors_per_cluster,
                       zoomed_in=True)
plt.savefig("results/figures/TRB/cluster_locations_IdFM_zoomed.pdf", bbox_inches='tight')

distances_stations_to_center, (mean_distance_cluster_to_center, std_distance_cluster_to_center) = dist

def scale_between(X, x1, x2, y1, y2):
    result = np.zeros_like(X)
    result[X < x1] = y1
    result[X > x2] = y2
    result[(X >= x1) * (X <= x2)] = ((X[(X >= x1) * (X <= x2)]-x1) / (x2 - x1)) * (y2-y1) + y1
    return result

sizes =  scale_between(distances_stations_to_center, 5000, 10000, 30, 180)


plot_clusters_location(log_r=log_r, id_array=ind_array, display_legends=False,
                              color_per_cluster=colors_per_cluster,
                              zoomed_in=False, sizes=sizes)
plt.savefig("results/figures/TRB/cluster_locations_IdFM_dezoomed.pdf", bbox_inches='tight')






#%%
distances_stations_to_center, (mean_distance_cluster_to_center, std_distance_cluster_to_center) = dist
plt.figure(figsize=(7, 15))

bins = np.array([0, 0.5, 1, 1.5, 3, 5, 10, 15, 30, 50, 100])  # in kilometers
n_bins = len(bins)-1

bin_centers = (bins[1:] + bins[:-1])/2
bin_widths = np.diff(bins)


for k in range(model.n_clusters):
    plt.subplot(model.n_clusters, 1, k+1)
    this_dist = distances_stations_to_center[chosen_cluster == k] /1000  # from meters to km
    this_dist = this_dist[~np.isnan(this_dist)]
    histogram, _ = np.histogram(this_dist, bins=bins)


    label = f'Cluster {k+1}\n({(chosen_cluster == k).sum()} stations)'
    plt.bar(x=np.arange(n_bins), width=1., bottom=0, height=histogram,
            label=label, facecolor=colors_per_cluster[k,:], edgecolor='k', zorder=+20)

    plt.grid(True)
    plt.legend(loc="upper left")
    if k != model.n_clusters-1:
        # plt.xticks([], [])
        plt.xticks(np.arange(n_bins+1)-0.5, [" "]*(n_bins+1))
    else:
        bins_rounded = [int(b) if int(b) == b else b for b in bins]
        plt.xticks(np.arange(n_bins+1)-0.5, bins_rounded)

plt.xlabel("Distance to the center of Paris (km)")
plt.savefig("results/figures/TRB/IdFM_distance_stations_to_center.pdf", bbox_inches='tight')



print("Distance of each cluster's stations to Paris:")
for k in range(model.n_clusters):
    print(f"\tcluster {k+1}: {mean_distance_cluster_to_center[k]/1000:.2f} +/- {std_distance_cluster_to_center[k]/1000:.2f} km ")










#%%

# =============================================================================
#            Difference in regression coefficients
# =============================================================================


i_time = variable_names["temporal"].index('time')


for s in range(model.n_segments):
    print(f"\nSegment {s+1}")
    for k in range(model.n_clusters):
        this_coef = model.beta[i_time,0,k,s]
        increase_all_interval = this_coef * (variables[1][0,:,i_time].max() - variables[1][0,:,i_time].min())
        increase_per_year = increase_all_interval * 1/5.5  # 5.5 years in the interval
        print(f"\tcluster {k+1}: {100*(10**increase_per_year -1):.1f}% per year")


#%%




variables_to_ignore  = ["strike_2019", "strike", "lockdown_1", "lockdown_2", "lockdown_3"]
variables_to_ignore += [vname for vname in variable_names["temporal"]  if ('spline' in vname)]

result = get_significant_differences(Y, variables, log_r, model, variable_names, print_vars=False,
                                    alpha=0.05, min_n_chosen=1, correction="Bonferroni",
                                    variables_to_ignore=variables_to_ignore,
                                    clusters_to_compare=[(n-1, n) for n in range(1,model.n_clusters)] )

(nonzeros_variables, difference_clusters, difference_segments) = result




chosen_vnames =  [vname for vname in variable_names["temporal"] if ("is_" in vname)] #+ ['school_holiday', 'national_holiday']
chosen_indices = [variable_names["temporal"].index(vname)  for vname in chosen_vnames ]


plt.figure(figsize=(4*(model.n_segments+1), 4))

subplot_list = []
for s in range(model.n_segments):
    subplot_list.append(plt.subplot(1,model.n_segments+1,s+1))
    plt.title(f"Segment {s+1}")

subplot_diff = plt.subplot(1,model.n_segments+1, model.n_segments+1)
plt.title(f"Difference between segments 1 and {model.n_segments}")
subplot_list.append(subplot_diff)
x_vars = np.arange(len(chosen_indices))

for k in range(model.n_clusters):
    for s in range(model.n_segments):

        var_contrib = model.beta[chosen_indices,0,k,s] #shape: (n_selected_vars,)
        var_contrib = 100* (10**(var_contrib)-1)# linear scale
        subplot_list[s].plot(x_vars, var_contrib, c=colors_per_cluster[k,:], zorder=k*10, lw=0.9)

        is_significant = np.array([(vname in nonzeros_variables[k][s]) for vname in chosen_vnames])
        subplot_list[s].scatter(x_vars[is_significant],  var_contrib[is_significant],
                                 marker='o', c=[colors_per_cluster[k,:]], zorder=k*10)
        subplot_list[s].scatter(x_vars[~is_significant], var_contrib[~is_significant],
                                 marker='x', c=[colors_per_cluster[k,:]], lw=0.7)
                                 # marker='o', edgecolor=[colors_per_cluster[k,:]],
                                 # facecolor=[[0., 0., 0., 0.]], zorder=k*10, lw=0.9)

    var_diff = model.beta[chosen_indices,0,k,model.n_segments-1] - model.beta[chosen_indices,0,k,0]
    var_diff = 100* (10**(var_diff)-1)# linear scale
    subplot_diff.plot(x_vars, var_diff, c=colors_per_cluster[k,:],
             label=f"cluster {k+1}", zorder=k*10, lw=0.9)

    is_significant = np.array([(vname in difference_segments[k][(0, model.n_segments-1)]) for vname in chosen_vnames])
    subplot_diff.scatter(x_vars[is_significant],  var_diff[is_significant],
                         marker='o', c=[colors_per_cluster[k,:]], zorder=k*10)
    subplot_diff.scatter(x_vars[~is_significant], var_diff[~is_significant],
                           marker='x', c=[colors_per_cluster[k,:]], zorder=k*10, lw=0.7)


for i, subplot in enumerate(subplot_list):
    subplot.set_xticks(np.arange(len(chosen_vnames)), [name[3:] for name in chosen_vnames], rotation=-30, ha="left")
    subplot.grid(True)
    subplot.set_ylabel("relative increase (%)")
    subplot.set_xlabel(f"({chr(ord('a')+i)})")


 # legend for the last subplot
subplot_diff.scatter([np.nan], [np.nan], marker='o', c=[[0., 0., 0.]], label="significant\n values")
subplot_diff.scatter([np.nan], [np.nan], marker='x', c=[[0., 0., 0.]], lw=0.9, label="insignificant\n values")

plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


# same scale for the first two subplots:
ymin = min([spt.get_ylim()[0] for spt in subplot_list])
ymax = max([spt.get_ylim()[1] for spt in subplot_list])


for spt in subplot_list[:-1]:
    spt.set_ylim(ymin, ymax)
plt.tight_layout()
plt.savefig("results/figures/TRB/IdFM_effect_variables.pdf", bbox_inches='tight')





#%%











plt.figure(figsize=(8, 8))

for k in range(model.n_clusters):

    for i_l in [1,2,3]:
        dx = 0.09

        index_lockdown = variable_names["temporal"].index(f"lockdown_{i_l}")
        coef_per_segment = model.beta[index_lockdown,0,k,:]   # shape: (n_segments,)
        is_lowckdown = variables[1][0,:,index_lockdown]  # shape: (n_days,)

        weights_segments_per_day = np.nansum(r[chosen_cluster==k,:,k,:], axis=0) # shape: (n_days, n_segments)
        weights_segments_per_day *= is_lowckdown.reshape(-1,1)
        weights_segments = np.nanmean(weights_segments_per_day, axis=0)
        weights_segments /= weights_segments.sum()
        weighted_coef = weights_segments @ coef_per_segment  # mean along days, we obtain a scalar

        variation = ((10**weighted_coef) - 1)*100

        label = f"cluster {k+1}" if i_l == 1 else None
        plt.bar(x=i_l+k*dx, width=dx, bottom=0, height=variation,
                facecolor=colors_per_cluster[k,:], label=label)


plt.ylabel("Variation (%)")

plt.xticks(np.arange(1,3+1) + (model.n_clusters/2)*dx, [f"Lockdown {i_l}" for i_l in [1,2,3]])
plt.legend()







#%%
