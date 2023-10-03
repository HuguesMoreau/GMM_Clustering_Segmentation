from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import datetime

from sklearn.preprocessing import SplineTransformer

from utils.pandas_utils import sort_dataframe_by
from preprocessing.IdFM.data_path import IdFM_path

IdFM_origin = np.datetime64('2015-01-01')
 # datetime_days = [IdFM_origin + datetime.timedelta(days=int(d)) for d in range(X.shape[1])]
dirnames = ['data-rf-2015',
            'data-rf-2016',
            'data-rf-2017',
            'data-rf-2018',
            'data-rf-2019',
            'data-rf-2020',
            'data-rf-2021',
            'data-rf-2022',]


rail_stations_filename = Path(r"auxiliary_data/emplacement-des-gares-idf-data-generalisee.csv")





#%%

def use_file(filename):
    valid_format = ((filename[-4:] == ".csv") or (filename[-4:] == ".txt"))
    number_entries = ("NB" in filename) or ("nombre-validations" in filename)
    return valid_format and number_entries



if __name__ == "__main__":

    used_dirnames = [dn for dn in dirnames if 'rf' in dn]
    dir_paths = [IdFM_path / Path(dirname) for dirname in used_dirnames]
    all_file_paths = [dir_path / Path(filename) for dir_path in dir_paths
                          for filename in os.listdir(dir_path) if use_file(filename)]
    all_file_paths.sort()

    #list_dataframes = [pd.read_csv(data_path + fname, sep=";") for fname in filenames]
    individual_dataframes = []
    for file_path in all_file_paths:
        file_extension = str(file_path)[-4:]
        sep = '\t|;'   # tab or semicolumn
        df = pd.read_csv(file_path, sep=sep,  encoding = 'unicode_escape',
                         engine='python')
        if 'ï»¿JOUR' in df.columns : # 2022 data is a bit problematic
            df = df.rename(columns={'ï»¿JOUR': 'JOUR'})
            
        try: 
            df["JOUR"] = pd.to_datetime(df["JOUR"], format="%Y-%m-%d")
        except ValueError:
            df["JOUR"] = pd.to_datetime(df["JOUR"], format="%d/%m/%Y")

        individual_dataframes.append(df)

    df = pd.concat(individual_dataframes)


    for field in df.columns:
        print(f"\n \t Column '{field}':")
        n_null =   df[field].isnull().sum()
        n_qmark = (df[field] == "ND").sum()
        n_ND =    (df[field] ==  "?").sum()
        n_zeros = (df[field] == '0' ).sum() + (df[field] == 0).sum()
        print(f"{n_null } Null values ({100* n_null/len(df):.2f} %)")
        print(f"{n_qmark} '?'  values ({100*n_qmark/len(df):.2f} %)")
        print(f"{ n_ND  } 'ND' values ({100*  n_ND /len(df):.2f} %)")
        sum_NaNs = n_null + n_qmark + n_ND
        print(f"In total, {sum_NaNs} unuseable values ({100* sum_NaNs/len(df):.2f} %)")

        print(f"{n_zeros} Zero values ({100*n_zeros/len(df):.2f} %)")



#%%


def harmonize_2017_rail_changes(df):
    """
    In 2017, several changes were applied:
        - several stations changed their indices ("ID_REFA_LDA"). These stations
            kept their names intact.
        - "CHATELET" station merged with "LES HALLES", giving "CHATELET-LES HALLES"
    This function reverses the changes and harmionizes the indices.

    Parameters
    ----------
    df: pandas Dataframe of the rail data
        In particular, df must have attributes 'ID_REFA_LDA', 'NB_VALD', and 'JOUR'

    Returns
    ----------
    df: pandas Dataframe
    """

    # Merging Chatelet with Les Halles
    station_chatelet_les_halles = df[df['LIBELLE_ARRET'] == "CHATELET-LES HALLES"]
    assert (station_chatelet_les_halles['ID_REFA_LDA'].nunique() == 1)

    id_chatelet_les_halles = station_chatelet_les_halles['ID_REFA_LDA'].unique()[0]
    separate_stations = df[df['LIBELLE_ARRET'].isin(["CHATELET", "LES HALLES"])].copy()
    separate_stations["ID_REFA_LDA"] = id_chatelet_les_halles
    separate_stations["LIBELLE_ARRET"] = "CHATELET-LES HALLES"
    first = lambda l:l.min()  # we expect the values to either 1) be equal or 2) not matter
    merge_functions = {'CODE_STIF_TRNS':first, 'CODE_STIF_RES':first, 'CODE_STIF_ARRET':first,
           'LIBELLE_ARRET':first, 'ID_REFA_LDA':first, 'NB_VALD':'sum'}
    merged_stations = separate_stations.groupby(["JOUR", "CATEGORIE_TITRE"]).aggregate(merge_functions)

    # remove old stations, add the new ones
    df = df[~df['LIBELLE_ARRET'].isin(["CHATELET", "LES HALLES"])]
    df = pd.concat([df, merged_stations.reset_index()])



    # alignment of IDs.
    # We could make a groupby on "CODE_STIF_ARRET", "JOUR", "CATEGORIE_TITRE", on the whole df,
    # but it would be too slow
    # all_ids = df['ID_REFA_LDA'].unique()
    all_stations = df.groupby(["CODE_STIF_ARRET"])  # CODE_STIF_ARRET is the ID that remains stable
        # we did not choose this as our ID because ID_REFA_LDA allows easier join with location data
    n_id_per_station = all_stations["ID_REFA_LDA"].nunique()
    names_to_merge = n_id_per_station[n_id_per_station >= 2].index

    to_merge = df["CODE_STIF_ARRET"].isin(names_to_merge)
    stations_to_merge = df[to_merge]

    del merge_functions['CODE_STIF_ARRET']
    stations_to_merge = stations_to_merge.groupby(["CODE_STIF_ARRET", "JOUR", "CATEGORIE_TITRE"]).aggregate(merge_functions)
    stations_to_merge = stations_to_merge.reset_index()

    df = df[~to_merge]
    df = pd.concat([df, merged_stations.reset_index()])

    return df




#%%

def load_transport_data(network_type, debug=False):
    """
    Parameters
    ----------
    network_type: string
        either 'surface' or 'rail'
    debug, optional: bool
        if True, triggers a test instead of loading the real data
        Defaults to False

    Returns
    -------
    Tuple (X, id_array, id_names, day_array, entry_types)
    X: np array with shape (n_individuals, n_days, n_entry_types)
    id_array: np int array with shape (n_individuals, )
        Contains the id of the 'CODE_STIF_ARRET'  attribute of the stop (rail network),
        or the 'CODE_STIF_LIGNE' attribute for the bus/tramway line.
        We consider the ID to be unique.
        To distonguish the rail network from the sirface one, we flip the sign of the
        surface IDs: a negative integer ID denotes a surface line, and a positive one
        denotes a rail stop.
    id_names: dict with
        keys = id (int)
        values = list of possible names (in the case of bus lines, two directions of the same line
                    have different names)
    day_array: np int array with shape (n_days,)
        Serves to make the correspondence between X and the temporal variables
        (matches with the 'day' attribute of the dataframe)
    entry_types: list of strings
        Provides information whether the entries come from one-use tickets or regular subscription
    """

    if network_type == 'rail':
        id_field_name = "ID_REFA_LDA"
        label_field = "LIBELLE_ARRET"
        directory_code = 'rf'
    elif network_type == 'surface':
        id_field_name = "CODE_STIF_LIGNE"
        label_field = "LIBELLE_LIGNE"
        directory_code = 'rs'
    else:
        raise ValueError(f"Unknown value for network_type, '{network_type}'. Available values are ['surface', 'rail']")


    used_dirnames = [dn for dn in dirnames if directory_code in dn]
    dir_paths = [IdFM_path / Path(dirname) for dirname in used_dirnames]
    all_file_paths = [dir_path / Path(filename) for dir_path in dir_paths
                          for filename in os.listdir(dir_path) if use_file(filename)]
    all_file_paths.sort()


    #list_dataframes = [pd.read_csv(data_path + fname, sep=";") for fname in filenames]
    individual_dataframes = []
    for file_path in all_file_paths:        
        sep = '\t|;'   # tab or semicolumn
        df = pd.read_csv(file_path, sep=sep,  encoding = 'unicode_escape',
                         engine='python')
        if 'ï»¿JOUR' in df.columns : # 2022 data is a bit problematic
            df = df.rename(columns={'ï»¿JOUR': 'JOUR'})
            
        try: 
            df["JOUR"] = pd.to_datetime(df["JOUR"], format="%Y-%m-%d")
        except ValueError:
            df["JOUR"] = pd.to_datetime(df["JOUR"], format="%d/%m/%Y")


        individual_dataframes.append(df)

    df = pd.concat(individual_dataframes)
    # columns of the dataframe
    # Rail: ['JOUR', 'CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_ARRET',
    #        'LIBELLE_ARRET', 'ID_REFA_LDA', 'CATEGORIE_TITRE', 'NB_VALD']
    # Surface: ['JOUR', 'CODE_STIF_TRNS', 'CODE_STIF_RES', 'CODE_STIF_LIGNE',
    #    'LIBELLE_LIGNE', 'ID_GROUPOFLINES', 'CATEGORIE_TITRE', 'NB_VALD']


    #     Data cleaning
    df = df[~(df[id_field_name].isin(["ND", " ND", "ND ", "   ", "?"]))]   # Remove rows where the stop/line is unknown
    df = df[~df[id_field_name].isna()]
    df[id_field_name] = df[id_field_name].astype(int)
    if network_type == 'rail': df = df[df[id_field_name] != 0]
    if network_type == 'surface': df[id_field_name] *= -1
    df["NB_VALD"].replace({"Moins de 5": 3}, inplace=True)
    df["NB_VALD"] = df["NB_VALD"].astype(int)
    df["CATEGORIE_TITRE"].replace({'?': 'AUTRE TITRE', 'NON DEFINI': 'AUTRE TITRE', }, inplace=True)
    if "LIBELLE_ARRET" in df.columns:
        df["LIBELLE_ARRET"].replace({"LUCIE AUBRAC  ":"LUCIE AUBRAC"}, inplace=True)


    if network_type == 'rail':
        df = harmonize_2017_rail_changes(df)

    dates_list = list(df["JOUR"].unique())
    dates_list.sort()
    delta_days = [(date-IdFM_origin).astype('timedelta64[D]') / np.timedelta64(1, 'D') for date in dates_list]
        # we need to recast a time difference as days because numpy would measure it in nanoseconds otherwise :/
    day_array = np.array(delta_days)
    n_days = len(day_array)

    id_array = df[id_field_name].unique().copy()
    id_names = {i:[] for i in id_array}

    n_individuals = df[id_field_name].nunique()
    entry_types = df["CATEGORIE_TITRE"].unique()
    dim = len(entry_types)  # we choose to include unknown ticket types
    """
        AMETHYSTE : handicapped/senior tariff
        Imagine R: schoolchild/student  (regular)
        TST: reduced-price solidarity tariff (Tarif solidarité Transport)
        FGT : free solidarity  (Forfait Gratuité Transport)
        NAVIGO: regular subscription
        NAVIGO JOUR: one-day ticket
    """



    X = np.zeros((n_individuals, n_days, dim), dtype=np.float32) -1
        # remaining negative values denote missing data

    for i_ind, ind_id in enumerate(id_array):
        selected_inds = df[df[id_field_name] == ind_id].copy()
        id_names[ind_id] = list(selected_inds[label_field].unique())

        for i_day, date in enumerate(dates_list):
            selected_date = selected_inds[selected_inds["JOUR"] == date].copy()

            for i_dim, entry_type in enumerate(entry_types):
                selected_entry_types = selected_date[selected_date['CATEGORIE_TITRE'] == entry_type]


                # known_exceptions = []
                # if entry_type != 'AUTRE TITRE':
                #     # qe = 3
                #     # assert len(selected_entry_types) < 2, f"More than one ({len(selected_entry_types)}) of the same value : code = '{ind_id}', date = '{date}', ticket type = '{entry_type}'"
                #     if len(selected_entry_types) > 1 and int(ind_id) != 689:
                #         print(f"More than one ({len(selected_entry_types)}) of the same value : code = '{ind_id}', date = '{date}', ticket type = '{entry_type}'")

                if len(selected_entry_types) > 0:
                    if debug:
                        X[i_ind, i_day, i_dim] += 1
                    else:
                        X[i_ind, i_day, i_dim] = selected_entry_types["NB_VALD"].sum()

    if debug:
        print("Debugging load_transport_data:")
        print(f"On {network_type} data: {100*(X < 0).mean():.2f}% missing values")
        print(f"(\t {100*((X < 0).all(axis=2)).mean():.2f}% missing individuals x days if we sum all the ticket types)")
        print(f"In addition, {1+delta_days[-1] - len(delta_days)} days are missing entirely (out of {delta_days[-1]})")


    return X, id_array, id_names, day_array, entry_types



if __name__ == "__main__":
    rail_data = load_transport_data('rail', debug=True)




#%%


def load_weather_Paris(day_array):
    """
    Parameters
    ----------
    day_array: np array of ints with shape (n_days,)
        each int is the number of days since 01/01/2015


    Returns
    -------
    weather_array: np.array with shape (n_days, 2)
        The columns are the mean temperature (°C) and total rainfall (mm)
    """


    with open(IdFM_path / Path("auxiliary_data/export-paris0.csv"), 'r') as f:
        weather_df = pd.read_csv(f, skiprows=3, header=0)

    weather_df["DATE"] = pd.to_datetime(weather_df["DATE"], format="%Y-%m-%d")

    weather_df["DAY_NO"] = (weather_df["DATE"] - IdFM_origin).astype('timedelta64[D]').astype(int)
    weather_df = sort_dataframe_by(weather_df, "DAY_NO", order=day_array)


    weather_df["Mean_temp_C"] = ( weather_df["TEMPERATURE_MORNING_C"]
                                + weather_df["TEMPERATURE_NOON_C"]
                                + weather_df["TEMPERATURE_EVENING_C"])/3

    return weather_df[["Mean_temp_C", "PRECIP_TOTAL_DAY_MM"]].to_numpy()


# with open(IdFM_path/Path("rail_data.pickle"), "wb") as f:
#     pickle.dump(rail_data, f)

if __name__ == "__main__":
    weather = load_weather_Paris(np.arange(6*365))


    plt.figure(figsize=(15, 15))
    ax_temp = plt.gca()
    ax_rain = ax_temp.twinx()
    ax_temp.plot(weather[:,0], c=[0.7, 0.0, 0.0], label="mean temperature")
    ax_temp.legend(loc="upper left")
    ax_rain.plot(weather[:,1], 'o', markersize=2, c=[0.0, 0.0, 0.6], label="total rainfall")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_rain.set_ylabel("rain (mm)")
    ax_rain.legend(loc="upper right")






#%%




def load_national_holidays(day_array):
    """
    Parameters
    ----------
    day_array: np array of ints with shape (n_days,)
        each int is the number of days since 01/01/2015


    Returns
    -------
    national_holiday_array: np.array of bools (n_days,)
    """


    with open(IdFM_path / Path("auxiliary_data/jours_feries_metropole.csv"), 'r') as f:
        bank_holiday_df = pd.read_csv(f, header=0, sep=",")

    bank_holiday_df["date"] = pd.to_datetime(bank_holiday_df["date"], format="%Y-%m-%d")
    bank_holiday_df["DAY_NO"] = (bank_holiday_df["date"] - IdFM_origin).astype('timedelta64[D]').astype(int)
    bank_holiday_df = sort_dataframe_by(bank_holiday_df, "DAY_NO", order=day_array)
        # contains NaNs fof non-holidays
    bank_holiday_df["national_holiday"] = ~ (bank_holiday_df["nom_jour_ferie"].isna())
                                                    # any field name will do

    return bank_holiday_df["national_holiday"].to_numpy().astype(bool)


if __name__ == "__main__":
    national_holidays = load_national_holidays(np.arange(6*365))
    datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in np.arange(6*365)]
    datetime_days = np.array(datetime_days)
    plt.figure()
    plt.plot(datetime_days[national_holidays],  national_holidays[national_holidays],  "o", c='r')
    plt.plot(datetime_days[~national_holidays], national_holidays[~national_holidays], "o", c='b')
    plt.plot(datetime_days, national_holidays, c=[0.5, 0.5, 0.5, 0.5])
    plt.title('is the day a national holiday ?')



    #%%

def load_strikes(day_array):
    """
    Parameters
    ----------
    day_array: np array of ints with shape (n_days,)
        each int is the number of days since 01/01/2015


    Returns
    -------
    strike_2019, other_strikes: two np.arrays of bools (n_days,)
    """


    from preprocessing.IdFM.RATP_strikes import start_end_strikes  # list of couple of dates
    datetime_array = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in day_array]
    datetime_array = np.array(datetime_array)


    strike_2019 = start_end_strikes[12]
    other_strikes = start_end_strikes[:12] + start_end_strikes[12+1:]  # remove but not in-place
    strike_2019_array = (datetime_array >= strike_2019[0]) & (datetime_array <= strike_2019[1])


    other_strikes_array = np.zeros_like(day_array).astype(bool)
    for (start, end) in other_strikes:
        this_strike = (datetime_array >= start) & (datetime_array <= end) # shape: (n_days,)
        other_strikes_array += this_strike

    return strike_2019_array, other_strikes_array



if __name__ == "__main__":
    strike_2019, other_strikes = load_strikes(np.arange(6*365))
    no_strike =  (~strike_2019) & (~other_strikes)
    datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in np.arange(6*365)]
    datetime_days = np.array(datetime_days)
    plt.figure()
    plt.plot(datetime_days[strike_2019],   strike_2019[strike_2019],     "o", c='r', label="end-2019")
    plt.plot(datetime_days[other_strikes], other_strikes[other_strikes], "o", c='b', label="other strikes")
    plt.plot(datetime_days[no_strike],     other_strikes[no_strike],         "o", c='g', label="no strike")
    plt.plot(datetime_days, ~no_strike, c=[0.5, 0.5, 0.5, 0.5])
    plt.title('is the day a strike ?')
    plt.legend()

#%%




def load_school_holidays(day_array):
    """
    Parameters
    ----------
    day_array: np array of ints with shape (n_days,)
        each int is the number of days since 01/01/2015


    Returns
    -------
    school_holiday_array: np.array of bools (n_days,)
    """


    from preprocessing.IdFM.school_holidays import start_end_school_holidays  # list of couple of dates
    datetime_array = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in day_array]
    datetime_array = np.array(datetime_array)

    school_holiday_array = np.zeros_like(day_array).astype(bool)
    for (start, end) in start_end_school_holidays:
        this_school_holiday = (datetime_array >= start) & (datetime_array <= end) # shape: (n_days,)
        school_holiday_array += this_school_holiday

    return school_holiday_array



if __name__ == "__main__":
    school_holidays = load_school_holidays(np.arange(6*365))
    datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in np.arange(6*365)]
    datetime_days = np.array(datetime_days)
    plt.figure()
    plt.plot(datetime_days[school_holidays],  school_holidays[school_holidays],  "o", c='r')
    plt.plot(datetime_days[~school_holidays], school_holidays[~school_holidays], "o", c='b')
    plt.plot(datetime_days, school_holidays, c=[0.5, 0.5, 0.5, 0.5])
    plt.title('is the day a school_holiday ?')


#%%



def get_rail_type(id_array):
    """
    Parameters
    ----------
    id_array: np array of ints with shape (n_individuals,)

    Returns
    -------
    RER_present:         np.array of bools with shape (n_individuals,)
    underground_present: np.array of bools with shape (n_individuals,)
    tramway_present:     np.array of bools with shape (n_individuals,)
    train_present:       np.array of bools with shape (n_individuals,)
    """

    df_station_types = pd.read_csv(IdFM_path / rail_stations_filename, sep = ";")
    df_station_types.rename(columns={"id_ref_ZdC":"id_ref_lda"}, inplace=True)  # some column names have changed
    df_station_types['id_ref_lda'] = df_station_types['id_ref_lda'].astype(float).astype(int)


    df_station_types["rer"] = df_station_types["rer"].astype(float)
    df_station_types["metro"] = df_station_types["metro"].astype(float)
    df_station_types["tramway"] = df_station_types["tramway"].astype(float)
    df_station_types["train"] = df_station_types["train"].astype(float)

    kept_station_types = df_station_types[df_station_types['id_ref_lda'].isin(id_array)]
    id_array_df = pd.DataFrame({"id_ref_lda":id_array})

    kept_station_types = id_array_df.merge(kept_station_types, on="id_ref_lda", how="left", validate="1:m")
    kept_station_types = sort_dataframe_by(kept_station_types, by='id_ref_lda', order=id_array)
    kept_station_types.drop_duplicates(subset='id_ref_lda', inplace=True)
    kept_station_types = kept_station_types[kept_station_types['id_ref_lda'].isin(id_array)]


    RER_present = kept_station_types["rer"].astype(float).to_numpy()
    underground_present = kept_station_types["metro"].astype(float).to_numpy()
    tramway_present = kept_station_types["tramway"].astype(float).to_numpy()
    train_present = kept_station_types["train"].astype(float).to_numpy()

    return RER_present, underground_present, tramway_present, train_present








#%%




def load_exogenous_variables_transport(chosen_variables, id_array, day_array):
    """

    Parameters
    ----------
    chosen_variables (optional): list of strings. Exemple of elements include
        For temporal variables:
            "weekend" 0 or 1 (float), depending if the day is among the first five days of the week or the last two.
            "summer_period" 0 or 1 (float).
            "strike" 0 or 1 (float).
            "strike_2019" 0 or 1 (float). A specific variable for the strike between December 2019 and January 2021.
            "school_holiday":  0 or 1 (float)
            "national_holiday": 0 or 1 (float), covers days like Christmas, New Year, end of WW2, etc.
            "lockdown" 0 or 1 (float). Only covers the three most strict periods, this does not take into
                considerations softer measures like early bar closings
            "lockdown_different": create three bool variables, one for each official lockdown.
            "day_of_week": create 7 bools, one for each day.
            "temperature" : average temperature of the day (°C)
            "temperature_squared" : the square of the temperature (°C²)
            "rain" : rainfall of the day (mm)
            "time": between 0 and 1.
            "year_splines_dX_nY": creates Y year-periodic splines of degree X with Y nodes per year.
        For individual variables:
            "RER_present": 1 if the individual is a station with a RER (fast train network) line,
                0 if it represents a bus line, an underground-only or tramway-only station
            "underground_present": same with underground (metro)
            "tramway_present": same with above-ground tramway
            "train_present": same with the national train system (defferent from RER)
            "rail_type": creates all four variables above
            "surface_network": 1 for bus and tramway lines, 0 for RER and underground stations.
        Note that all variable names are in the same list, the function will know which is which



    Returns
    -------
    individual_variables: np array with shape (n_individuals, 1,      n_ind_variables)
    temporal_variables:   np array with shape (1,             n_days, n_temp_variables)
    mixed_variables:      np array with shape (n_individuals, n_days, n_mixed_variables)
        Note that these arrays may have zero columns if no variable is chosen.
        Currently, this is at least the case for mixed variables.
    variable_names: dict with
        keys: "individual", "temporal" and "mixed"
        values: list of strings
            the order of the strings matches the order of the columns in the variable matrices.



    The order of the individuals (resp. days) is such that
    for all i in range(n_individuals), individual_variables[i,:]
    gives the variable of the station with ID given in id_array[i]

    When there are missing values, these values are replaced by NaNs.
    """

    variable_names = {"individual":[], "temporal":[], "mixed":[]}

    # =============================================================================
    #                    Individual-specific variables
    # =============================================================================
    possible_ind_variables = ["surface_network", "RER_present", "underground_present", "tramway_present",
                              "train_present", "rail_type"]
            # "RER_present": 1 if the individual is a station with a RER (fast train network) line,
            #     0 if it represents a bus line or an underground-only station
    chosen_ind_variables = [vname for vname in chosen_variables if vname in possible_ind_variables]
    n_i_variables = len(chosen_ind_variables)
    n_i_variables += (4-1) if "rail_type" in chosen_ind_variables else 0
    individual_variables = np.empty((id_array.shape[0], 1, n_i_variables), dtype=np.float32)
    variable_names["individual"] = []  # will be updated later
    index_ind_var = 0

    if "surface_network" in chosen_ind_variables:
        individual_variables[:,0,index_ind_var] = (id_array < 0)
        index_ind_var += 1
        variable_names["individual"].append("surface_network")


    if "RER_present" in chosen_ind_variables:
        RER_present, _ = get_rail_type(id_array)
        individual_variables[:,0,index_ind_var] = RER_present
        index_ind_var += 1
        variable_names["individual"].append("RER_present")


    if "underground_present" in chosen_ind_variables:
        _, underground_present = get_rail_type(id_array)
        individual_variables[:,0,index_ind_var] = underground_present
        index_ind_var += 1
        variable_names["individual"].append("underground_present")

    if "tramway_present" in chosen_ind_variables:
        _,_, tramway_present, _ = get_rail_type(id_array)
        individual_variables[:,index_ind_var] = tramway_present
        index_ind_var += 1
        variable_names["individual"].append("tramway_present")

    if "train_present" in chosen_ind_variables:
        _, _, _, train_present = get_rail_type(id_array)
        individual_variables[:,0,index_ind_var] = train_present
        index_ind_var += 1
        variable_names["individual"].append("train_present")


    if "rail_type" in chosen_ind_variables:
        RER_present, underground_present, tramway_present, train_present = get_rail_type(id_array)
        individual_variables[:,0,index_ind_var+0] = RER_present
        individual_variables[:,0,index_ind_var+1] = underground_present
        individual_variables[:,0,index_ind_var+2] = tramway_present
        individual_variables[:,0,index_ind_var+3] = train_present
        index_ind_var += 4
        variable_names["individual"] += ["RER_present", "underground_present", "tramway_present", "train_present"]


    # =============================================================================
    #     Temporal variables
    # =============================================================================
    possible_temp_variables = ["weekend", "temperature", "rain", "temperature_squared", "summer_period", "day_of_week", "strike", "strike_2019",
                               "lockdown", "lockdown_different", "time", "national_holiday", "school_holiday" ]  # splines excluded
    chosen_temp_variables = [vname for vname in chosen_variables if vname in possible_temp_variables]
    n_t_variables = len(chosen_temp_variables)
    n_t_variables += (7-1) if "day_of_week" in chosen_temp_variables else 0
    n_t_variables += (3-1) if "lockdown_different" in chosen_temp_variables else 0

    # add the splines now
    spline_vnames = [vname for vname in chosen_variables if "year_splines_" in vname]  # should be either [] or ["year_splines_dX_nY"]
    if len(spline_vnames) > 0:
        assert len(spline_vnames) == 1, "You may only specify one set of splines"
        spline_vname = spline_vnames[0]
        string_elements = spline_vname.split("_")  # list, string_elements[0] = "year", string_elements[1] = "splines"
        spline_degree_str = string_elements[2] # "dX"
        spline_degree     = int(spline_degree_str[1:])
        spline_n_knots_str = string_elements[3] # "nY"
        spline_n_knots     = int(spline_n_knots_str[1:])

        n_splines = spline_n_knots - 1  # we will set include_bias to True and deal with the indetermination
             # during the regression
        n_t_variables += n_splines  # one spline per node (bias is included elsewhere)
        chosen_temp_variables.append("year_splines")

    temporal_variables = np.empty((1, day_array.shape[0], n_t_variables), dtype=np.float32)
    index_temp_var = 0 # where to insert the next variable

    if "weekend" in chosen_variables:
        # we convert the day integers to datatime object to get the corresponding day of the week
        # this seems convoluted, but it avoids mistakes.
        datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in day_array]
        day_of_week = np.array([d.weekday() for d in datetime_days])  # Monday is 0 and Sunday is 6
        weekend = (day_of_week >= 5).astype(np.float)
        temporal_variables[0,:,index_temp_var] = weekend
        index_temp_var += 1
        variable_names["temporal"].append("weekend")


    if "day_of_week" in chosen_variables:
        datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in day_array]
        day_of_week = np.array([d.weekday() for d in datetime_days])  # Monday is 0 and Sunday is 6
        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i in range(7):
            temporal_variables[0,:,index_temp_var] = (day_of_week == i)
            index_temp_var += 1
            variable_names["temporal"].append(f"is_{day_name[i]}")



    if "summer_period" in chosen_variables:
        datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in day_array]
        # summer_periods = [(datetime.datetime(i, 7, 1), datetime.datetime(i, 9, 1)) for i in range(2010, 2025)]
        summer_period = [(d.month >= 7) & (d.month < 9) for d in datetime_days]
        temporal_variables[0,:,index_temp_var] = np.array(summer_period, dtype=bool)
        index_temp_var += 1
        variable_names["temporal"].append("summer_period")


    if "school_holiday" in chosen_variables:
        school_holidays = load_school_holidays(day_array)
        temporal_variables[0,:,index_temp_var] = school_holidays.astype(float)
        index_temp_var += 1
        variable_names["temporal"].append("school_holiday")


    if "national_holiday" in chosen_variables:
        national_holidays = load_national_holidays(day_array)
        temporal_variables[0,:,index_temp_var] = national_holidays.astype(float)
        index_temp_var += 1
        variable_names["temporal"].append("national_holiday")


    if "strike" in chosen_variables:
        strike_2019, other_strikes = load_strikes(day_array)

        if "strike_2019" in chosen_variables:
            strikes = other_strikes
        else:
            strikes = strike_2019 + other_strikes
        temporal_variables[0,:,index_temp_var] = strikes.astype(float)
        index_temp_var += 1
        variable_names["temporal"].append("strike")


    if "strike_2019" in chosen_variables:
        strike_2019, _ = load_strikes(day_array)

        temporal_variables[0,:,index_temp_var] = strike_2019.astype(float)
        index_temp_var += 1
        variable_names["temporal"].append("strike_2019")



    if "time" in chosen_variables:
        temporal_variables[0,:,index_temp_var] = np.linspace(0, 1, day_array.shape[0])
        index_temp_var += 1
        variable_names["temporal"].append("time")



    lockdowns_list = [(datetime.date(2020,  3, 17), datetime.date(2020,  5, 11)),
                        (datetime.date(2020, 10, 30), datetime.date(2020, 12, 15)),
                        (datetime.date(2021,  4,  4), datetime.date(2021,  5,  3))]
    if "lockdown" in chosen_variables:
        datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in day_array]
        datetime_days = np.array(datetime_days)
        lockdown = np.zeros_like(day_array).astype(np.float32)
        for (start, end) in lockdowns_list:
            lockdown += (datetime_days >= start) & (datetime_days < end)
        temporal_variables[0,:,index_temp_var] = lockdown.astype(float)
        index_temp_var += 1
        variable_names["temporal"].append("lockdown")


    if "lockdown_different" in chosen_variables:
        datetime_days = [IdFM_origin.astype('M8[D]').astype('O') + datetime.timedelta(days=int(d)) for d in day_array]
        datetime_days = np.array(datetime_days)

        for i, (start, end) in enumerate(lockdowns_list):
            lockdown = (datetime_days >= start) & (datetime_days <= end)  # first lockdown
            temporal_variables[0,:,index_temp_var] = lockdown.astype(float)
            index_temp_var += 1
            variable_names["temporal"].append(f"lockdown_{i+1}")



    if ("temperature" in chosen_variables) or ("rain" in chosen_variables) or ("temperature_squared" in chosen_variables):
        weather_array = load_weather_Paris(day_array)
        if "temperature" in chosen_variables:
            temporal_variables[0,:,index_temp_var] = weather_array[:,0]
            index_temp_var += 1
            variable_names["temporal"].append("temperature")
        if "rain" in chosen_variables:
            temporal_variables[0,:,index_temp_var] = weather_array[:,1]
            index_temp_var += 1
            variable_names["temporal"].append("rain")
        if "temperature_squared" in chosen_variables:
            temporal_variables[0,:,index_temp_var] = weather_array[:,0]**2
            index_temp_var += 1
            variable_names["temporal"].append("temperature_squared")




    if "year_splines" in chosen_temp_variables:
        # we defined variables spline_n_knots, spline_degree, and n_splines previously
        spt = SplineTransformer(n_knots=spline_n_knots, degree=spline_degree,
                                extrapolation="periodic", include_bias=True,  # as mentionned, we will deal with the indetermination during the regression
                                knots=np.linspace(0,365,spline_n_knots).reshape(-1,1))
        splines = spt.fit_transform(day_array.reshape(-1,1))  # shape: (n_days, n_splines)
        temporal_variables[0,:,index_temp_var:index_temp_var+n_splines] = splines
        index_temp_var += n_splines
        for i in range(n_splines):
            argmax_index = np.argmax(splines[:,i])
            peak_date = IdFM_origin.tolist() + datetime.timedelta(days=day_array[argmax_index])
            spline_name = f"year_spline_{i:02} (max:{peak_date.strftime(' %b %d')})" # we assume there will be less than 101 splines
            variable_names["temporal"].append(spline_name)





    # =============================================================================
    #                        no mixed variables
    # =============================================================================
    mixed_variables = np.zeros((id_array.shape[0], day_array.shape[0], 0))

    assert individual_variables.shape[0] ==  id_array.shape[0]
    assert temporal_variables.shape[1]   == day_array.shape[0]

    return individual_variables, temporal_variables, mixed_variables, variable_names
















