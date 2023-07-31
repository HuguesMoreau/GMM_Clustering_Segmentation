"""
However, given the changes that occurred in the database indexation and recording protocol before 2017, we focus only on the five and a half years between January 1, 2017, to June 30, 2022
"""

from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import datetime


from preprocessing.IdFM.data_path import IdFM_path
from preprocessing.IdFM.load_raw_data import load_exogenous_variables_transport, load_transport_data, IdFM_origin, load_strikes













#%%
def load_data(network_type:str, ticket_types:str='all_distinct', start_date=None, end_date=None,
              sampling=None, chosen_variables=[], normalisation=None, remove_mean=None, scale='linear', verbose=False):
    """
    Parameters
    ----------
    network_type: either "rail", "surface", or "both"
    ticket_types: string, optional. How the different ticket types are distinguished in the data. Must be one of:
        "all_distinct" (default): all seven types of tickets are encoded as separate dimensions:
            AMETHYSTE (handicapped/senior tariff), Imagine R (schoolchild/student), TST (reduced-price solidarity tariff),
            FGT (free solidarity subscription), NAVIGO (regular subscription), NAVIGO JOUR (one-day ticket), AUTRE (other)
        "regularity" Distinguishes the one-trip tickets (NAVIGO JOUR, AUTRE) from passes used byregular travellers (AMETHYSTE, Imagine R, TST, FGT, NAVIGO)
            To make sure that the "other" category contains only single-use tickets, see all ticket types
            at https://data.iledefrance-mobilites.fr/explore/dataset/titres-et-tarifs (access date 01/04/2023),
            and check that the only ticket types which do not already appear are single-use tickets.
        "merge": all ticket types are summed into one.
    start_date: a datetime.datetime object, optional
        If omitted, start at the beginning of the dataset
    end_date: a datetime.datetime object, optional (in which case we go to the end)
    sampling (optional): list, which elements are couple of floats.
        For a couple (of floats), each float is the proportion of individuals (first element) and
            days (second element) we will keep at random
        If one of the elements of a couple is None, this means that all days/individuals will be used.
            For example, sampling = [(0.8, None), (0.2, None)] splits the individuals 80/20, and
            puts all days in each subset.
        The subsets are mutually exclusive and may not encompass all the data.
            For instance, if we set sampling to [(0.5, 0.5), (0.5, 0.5)], the function will create
            two subsets, each subset will contain 50% of days for 50 % of individuals, i.e. 25 % of the data.
            The remaining 50% of the whole dataset (the days of set 1 in the individuals of set 2 plus the
            days of set 2 in the indicviduals of set 1) will be discarded.
    chosen_variables (optional): list of strings. Exemple of elements include
        For temporal variables:
            "weekend" 0 or 1 (float), depending if the day is among the first five days of the week or the last two.
            "summer_period" 0 or 1 (float)
            "school_holiday":  0 or 1 (float)
            "national_holiday": 0 or 1 (float), covers days like Christmas, New Year, end of WW2, etc.
            "strike" 0 or 1 (float).
            "strike_2019" 0 or 1 (float). A specific variable for the strike between December 2019 and January 2021.
            "lockdown" 0 or 1 (float). Only covers the three most strict periods, this does not take into
                considerations softer measures like early bar closings
            "lockdown_different": create three bool variables, one for each official lockdown.
            "day_of_week": create 7 bools, one for each day.
            "temperature" : average temperature of the day (°C)
            "temperature_squared" : the square of the temperature (°C²)
            "rain" : rainfall of the day (mm)
            "time": between 0 and 1.
            "year_splines_dA_nB": creates B year-periodic splines of degree A with B nodes per year.
        For individual variables:
            "RER_present": 1 if the individual is a station with a RER (fast train network) line,
                0 if it represents a bus line, an underground-only or tramway-only station
            "underground_present": same with underground (metro)
            "tramway_present": same with above-ground tramway
            "train_present": same with the national train system (different from RER)
            "rail_type": creates all four variables above
            "surface_network": 1 for bus and tramway lines, 0 for RER and underground stations.
        No mixed variables are available
        Note that all variable names are in the same list, the function will know which is which
    normalisation (optional): string, one of "day_individual", "individual", "none"
        defaults to 'none'.
        Note that normalization divides by the mean instead of standard deviation.
    remove_mean (optional): string, one of "day_individual", "individual", "none"
        defaults to 'none'
    scale (string, optional): either 'linear', 'log', or 'log10'.
        defaults to 'linear'. To avoid extreme  negative values when normalizing, the lower values
        of the consumption are clipped to the 1st percentile consumptions.
    verbose: bool, defaults to False
        Whether to priint some values about the preprocessing



    Returns
    -------
    a couple (subset_list, variable_names)
    data: a list of 5-tuples. Each tuple contains:
        Y: np array with shape (n_individuals, n_days, dim)
            the dimension depends onthe value of ticket_types in the input:
            "all_distinct": 7 (in order: AMETHYSTE, AUTRE TITRE, FGT, IMAGINE R, NAVIGO, TST, NAVIGO JOUR,
            "regularity":   2 (travel cards with subscription vs. single-use or daily-use tickets),
            "merge":        1
        id_array:  np int array with shape (n_individuals, )
        day_array: np int array with shape (n_days,)
        variables: triple of:
            ind_variables:   np array with shape (n_individuals, 1,      n_ind_variables)
            temp_variables:  np array with shape (1,             n_days, n_temp_variables)
            mixed_variables: np array with shape (n_individuals, n_days, n_mixed_variables)
            Note that these arrays may have zero columns if no individual/temporal variable is chosen
            /!\ the exogenous variables are not standard-scaled, even if some normalisation has been specifiec on the data
    variable_names: dict with
        keys: "individual", "temporal" and "mixed"
        values: list of strings
            the order of the strings matches the order of the columns in the variable matrices.
    id_names: dict with
        keys = id (int)
        values = list of possible names (in the case of bus lines, two directions of the same line
                    have different names)
    time_origin: a datetime.datetime object
        Mostly usefun so that

    """

    error_msg = f"unknown value for network_type, '{network_type}'. Allowed values are 'surface', 'rail', 'both'"
    assert network_type in ["rail", "surface", "both"], error_msg



    if network_type in ["surface", "both"]:
        if 'surface_data.pickle' in os.listdir(IdFM_path):
            with open(IdFM_path/Path("surface_data.pickle"), "rb") as f:
                 surface_data = pickle.load(f)
        else:
            surface_data = load_transport_data('surface', debug=False)
            with open(IdFM_path/Path("surface_data.pickle"), "wb") as f:
                 pickle.dump(surface_data, f)

    if network_type in ["rail", "both"]:
        if 'rail_data.pickle' in os.listdir(IdFM_path):
            with open(IdFM_path/Path("rail_data.pickle"), "rb") as f:
                 rail_data = pickle.load(f)
        else:
            rail_data = load_transport_data('rail', debug=False)
            with open(IdFM_path/Path("rail_data.pickle"), "wb") as f:
                 pickle.dump(rail_data, f)

    if network_type == "rail":
        Y, id_array, id_names, day_array, entry_types = rail_data
        del rail_data
    elif network_type == "surface":
        Y, id_array, id_names, day_array, entry_types = surface_data
        del surface_data
    else: #  network_type ==  "both"
        assert rail_data[3].shape == surface_data[3].shape, "different number of days between rail and surface data"
        assert (rail_data[3] == surface_data).all(),                  "different days between rail and surface data"
        day_array = rail_data[3]

        entry_types_rail, entry_types_surface = rail_data[-1], surface_data[-1]
        if entry_types_rail != entry_types_surface:  # we care mostly about order
            rail_order    = np.argsort(entry_types_rail)
            surface_order = np.argsort(entry_types_surface)
            assert entry_types_rail[rail_order] == entry_types_surface[surface_order]
            Y_rail =       rail_data[0][:,:,rail_order]
            Y_surface = surface_data[0][:,:,surface_order]
            Y = np.concatenate([Y_rail, Y_surface], axis=0)
            entry_types = entry_types_rail[rail_order]
        else:
            Y = np.concatenate([rail_data[0], surface_data[0]], axis=0)
            entry_types = rail_data[-1]
        id_array = np.concatenate([rail_data[1], surface_data[1]], axis=0)
        id_names = rail_data[2] + surface_data[2]  # these are dicts
        del rail_data, surface_data

    if verbose:
        print(f"\nRaw data: {Y.shape[0]} stations")

    individual_variables, temporal_variables, mixed_variables, variable_names = load_exogenous_variables_transport(chosen_variables, id_array, day_array)

    if start_date is None:
        time_origin = IdFM_origin.tolist()  # cast as datetime.datetime
        start_day_ind = 0
    else:
        time_origin = start_date
        start_day_ind = (start_date - IdFM_origin.tolist()).days
        assert (start_day_ind >= 0) & (start_day_ind <= Y.shape[1]), f"incorrect starting date provided: {start_date.strftime('%Y-%m-%d')}"

    if end_date is None:
        end_day_ind = -1
    else:
        end_day_ind = (end_date - IdFM_origin.tolist()).days  - start_day_ind
        assert (end_day_ind >= 0) & (end_day_ind <= Y.shape[1]), f"incorrect end date provided: {end_date.strftime('%Y-%m-%d')}"



    Y = Y[:,start_day_ind:end_day_ind,:].astype(np.float64)
    temporal_variables = temporal_variables[:,start_day_ind:end_day_ind,:]
    mixed_variables    =    mixed_variables[:,start_day_ind:end_day_ind,:]
    day_array = day_array[start_day_ind:end_day_ind]

    n_individuals, n_days = id_array.shape[0], day_array.shape[0]




    #dealing with empty values
    min_people_per_day = 500
    max_missing_days = 0.6  # proportion of the time interval
    all_empty = (Y < 0).all(axis=2) # shape: (n_individuals, n_days)  all ticket types are missing
    proportion_missing_per_ind = all_empty.mean(axis=1)   # shape: (n_individuals,)
    mean_vals = np.nansum(Y.sum(axis=2), axis=1) /n_days  # shape: (n_individuals,)
            # we do not use np.nansum because a missing day might be zero here. We will use nanmean everywhere else
    kept_ind = (proportion_missing_per_ind < max_missing_days) & (mean_vals > min_people_per_day)
    Y = Y[kept_ind,:,:]

    if verbose:
        print(f"\t {(proportion_missing_per_ind > max_missing_days).sum()} stations have more than {100*max_missing_days:.1f} % missing days and were removed")
        print(f"\t {(mean_vals < min_people_per_day).sum()} stations have less than {min_people_per_day} entries per day and were removed")
        print("\t  (some stations can be in both categories)")
        print(f"\t{kept_ind.sum()} stations remain \n")

    all_empty = all_empty[kept_ind,:]
    id_array = id_array[kept_ind]
    individual_variables = individual_variables[kept_ind,:,:]
    mixed_variables      =      mixed_variables[kept_ind,:,:]
    n_individuals = np.nansum(kept_ind)

    Y[all_empty,:] = np.nan
    Y[Y<0] = 0  # nan < 0 == False, so this means if only some type of tichets are mossing, we ignore them
        # the case might happen, e.g. because some type of fares appeared in 2017
    # Henceforth, NaN values in Y denotes missing (day x individuals)





    # Possible merging of tickets types
    if ticket_types == "regularity":
        i_daily = list(entry_types).index("NAVIGO JOUR")
        i_other = list(entry_types).index("AUTRE TITRE")
        if verbose:
            print("\nRegularity of the ticket typess")
            print(entry_types)
            print("daily", i_daily)
            print("other", i_other)
        single_use = (np.arange(len(entry_types)) == i_daily) | (np.arange(len(entry_types)) == i_other)   # shape: (7,)
        Y_by_regularity = np.zeros((n_individuals, n_days, 2))
        Y_by_regularity[:,:,0] = np.sum(Y[:,:,~single_use ], axis=2)  # fares for consistent use
        Y_by_regularity[:,:,1] = np.sum(Y[:,:, single_use ], axis=2)  # one-use or day tickets for casual use.
        Y = Y_by_regularity

    elif ticket_types == "merge":
        Y = np.clip(Y, 0, np.inf)
        Y = np.sum(Y, axis=2, keepdims=True)  # shape: (n_individuals, n_days, 1)
    else:
        error_msg = f"unknown value for ticket_types, '{ticket_types}'. Allowed values are 'all_distinct', 'regularity', 'merge'"
        assert ticket_types == "all_distinct", error_msg





    # =============================================================================
    #       Standardisation, scaling
    # =============================================================================

    first_year = (np.arange(n_days) < 365)  # shape: (n_days,)
    strikes = load_strikes(day_array)  # couple of arrays with shape: (n_days,)
    first_year[(strikes[0] > 0) | (strikes[1] > 0)] = False

    if (normalisation is None) or normalisation == "none":
        Y = Y
    else:  # we will run into problems if
        if normalisation == "individual":
            Y /= (np.nanmean(Y[:,first_year,:], axis=1, keepdims=True) + 1e-8)  # values are integers, but we can never be sure
        elif normalisation == "day_individual":
            Y /= (np.nanmean(Y[:,first_year,:], axis=2, keepdims=True) + 1e-8)
        else:
            raise ValueError(f"Unknown normalisation '{normalisation}'. Must be one of 'day_individual', 'individual', 'none'.")




    if (remove_mean is None) or (remove_mean == "none") or (remove_mean == False):
        Y = Y  # we could have checked this condition before raising an error, but I find this cleaner
    elif remove_mean == "individual":
        Y -= np.nanmean(Y[:,first_year,:], axis=1, keepdims=True)
    elif remove_mean == "day_individual":
        Y -= np.nanmean(Y[:,first_year,:], axis=2, keepdims=True)
    else:
        raise ValueError(f"Unknown remove_mean argument '{remove_mean}'. Must be one of 'day_individual', 'individual', 'none'.")



    assert scale in ['linear', 'log', 'log10'], f"Unknown value for the scale parameter:'{scale}'. Use either 'linear', 'log', or 'log10'"
    if scale == 'log':
        Y = np.log(Y)
    elif scale == 'log10':
        Y = np.log(Y) / np.log(10)


    n_nan_before_filtering = np.isnan(Y).sum()
    Y_no_nan = Y.copy()
    Y_no_nan[np.isnan(Y)] = 0
    filter_mean = np.ones(15) / 15
    if (scale == 'log') or (scale == 'log10'):
        threshold = 1. if (scale == 'log10') else 2.  # about 10%
        for i in range(n_individuals):
            moving_avg =  np.convolve(Y_no_nan[i,:,0], filter_mean, 'same')
            Y[i,:,0][np.real(Y_no_nan[i,:,0]) < moving_avg - threshold] = np.nan
    else: # scale == "linear"
        for i in range(n_individuals):
            moving_avg =  np.convolve(Y_no_nan[i,:,0], filter_mean, 'same')
            Y[i,:,0][np.real(Y_no_nan[i,:,0]) < moving_avg / 10] = np.nan

    n_nan_after_filtering = np.isnan(Y).sum()
    n_filtered_values = n_nan_after_filtering - n_nan_before_filtering
    if verbose:
        print(f'\nFiltering removed the values of {n_filtered_values} days ({100*n_filtered_values/Y.size:.1f} % of days * individuals)')
        print(f'There are now {n_nan_after_filtering} missing days in the data ({100*n_nan_after_filtering/Y.size:.1f} %)')
        print("")








    # =============================================================================
    #         Sampling
    # =============================================================================
    if sampling is None:
        data = [(Y, id_array, day_array, (individual_variables, temporal_variables, mixed_variables))]
        kept_masks_individuals = [np.ones(n_individuals, dtype=bool)]
        kept_masks_days        = [np.ones(n_days,        dtype=bool)]

    else: # sampling is a list. For each subset, we generate two boolean masks
        #  to know which individuals (resp. days) belong in the subset
        n_subsets = len(sampling)
        # sampling of individuals
        if np.any([ind_sampling is None for (ind_sampling, day_sampling) in sampling]):
            assert np.all([ind_sampling is None for (ind_sampling, day_sampling) in sampling])
            kept_masks_individuals = [np.ones(n_individuals, dtype=bool)] * n_subsets
        else:# create a partition of the individuals
            subset_proportions = np.array([0]+[ind_sampling for (ind_sampling, day_sampling) in sampling])  # shape: (n_subsets+1,) (we added a zero at the beginning)
            subset_cum_proportions = np.cumsum(subset_proportions)   # shape: (n_subsets+1,)
            assert ((subset_proportions >= 0).all() and subset_cum_proportions[-1] <= 1.), "we allow to take less individuals than the total, but we cannot take more"
            subset_thresholds_ind = np.round(subset_cum_proportions * n_individuals).astype(int)   # shape: (n_subsets+1,)
            individuals = np.arange(n_individuals)
            individuals_shuffled = individuals.copy()
            np.random.shuffle(individuals_shuffled)
            kept_masks_individuals = []
            for i_subset in range(n_subsets):
                start, stop = subset_thresholds_ind[i_subset], subset_thresholds_ind[i_subset+1]
                    # reminder: following the definition of subset_proportions, there is a [0] at the start of subset_thresholds
                kept_individuals = individuals_shuffled[start:stop]
                this_mask = np.isin(individuals, kept_individuals)
                kept_masks_individuals.append(this_mask)

        # sampling of days
        if np.any([day_sampling is None for (ind_sampling, day_sampling) in sampling]):
            assert np.all([day_sampling is None for (ind_sampling, day_sampling) in sampling])
            kept_masks_days = [np.ones(n_days, dtype=bool)] * n_subsets
        else:
            subset_proportions = np.array([0]+[day_sampling for (ind_sampling, day_sampling) in sampling])  # shape: (n_subsets+1,) (we added a zero at the beginning)
            subset_cum_proportions = np.cumsum(subset_proportions)   # shape: (n_subsets+1,)
            assert ((subset_proportions >= 0).all() and subset_cum_proportions[-1] <= 1.), "we allow to take less days than the total, but we cannot take more"
            subset_thresholds_day = np.round(subset_cum_proportions * n_days).astype(int)   # shape: (n_subsets+1,)
            days = np.arange(n_days)
                 # we do not shuffle the days because we want a time-consistent partition, which improves
                 # the independance of the train eand test sets.
            kept_masks_days = []
            for i_subset in range(n_subsets):
                start, stop = subset_thresholds_day[i_subset], subset_thresholds_day[i_subset+1]
                    # reminder: following the definition of subset_proportions, there is a [0] at the start of subset_thresholds
                kept_days = days[start:stop]
                this_mask = np.isin(days, kept_days)
                kept_masks_days.append(this_mask)


        data = []
        for i_subset in range(n_subsets):
            mask_ind  = kept_masks_individuals[i_subset]
            mask_days = kept_masks_days[i_subset]
            Y[mask_ind,:,:]
            Y[mask_ind,:,:][:,mask_days,:]
            this_subset_data = (Y[mask_ind,:,:][:,mask_days,:],
                                id_array[ mask_ind],
                                day_array[mask_days],
                                (individual_variables[mask_ind,:,:],
                                 temporal_variables[:,mask_days,:],
                                 mixed_variables[mask_ind,:,:][:,mask_days,:]))
            data.append(this_subset_data)
        del Y

    return data, variable_names, id_names, time_origin



if __name__ =="__main__":
    chosen_variables = ['school_holiday', 'rain']
    data, variable_names, id_names = load_data("rail",    chosen_variables=chosen_variables)
    data, variable_names, id_names = load_data("surface", chosen_variables=chosen_variables)

    #%%
    data, variable_names = load_data("surface",    chosen_variables=chosen_variables)

    Y = data[0][0]
    plt.figure()
    # plt.imshow((Y < 0).all(axis=2), cmap='binary')

    some_but_not_all = ((Y < 0).any(axis=2)) & (~ (Y < 0).all(axis=2))
    plt.imshow(some_but_not_all, cmap='binary')
    datetime_days = [datetime.datetime(year=2015, month=1, day=1) + datetime.timedelta(days=int(d)) for d in range(Y.shape[1])]
    plt.xticks(np.arange(0, Y.shape[1], 181), datetime_days[::181], rotation=-45)

    # load_data("both", sampling=[(0.5, None), (0.2, None)], chosen_variables=chosen_variables)
    # load_data("both", sampling=[(None, 0.1), (None, 0.3)], chosen_variables=chosen_variables)
    # load_data("both", sampling=[(0.2, 0.1),  (0.4,  0.7)], chosen_variables=chosen_variables)
    # load_data("both", sampling=[(0.2, 0.1),  (0.4,  0.7)], chosen_variables=chosen_variables)

    # load_data("rail",    chosen_variables=chosen_variables)
    # load_data("surface", chosen_variables=chosen_variables)
    # load_data("both",    chosen_variables=chosen_variables)



    # load_data("both", ticket_types="regularity", chosen_variables=chosen_variables)
    # load_data("both", ticket_types="merge", chosen_variables=chosen_variables)
    # load_data("both", ticket_types="all_distinct", chosen_variables=chosen_variables)




