"""
The code here serves to plot the clusters of IdFM stations on a map.

It also serves to compute the average distance to Paris city.
"""

from pathlib import Path
import numpy as np
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt

import geopandas
import contextily as ctx

from preprocessing.IdFM.data_path import IdFM_path
from utils.pandas_utils import sort_dataframe_by
from utils.nan_operators import nan_logsumexp

mercator = 'EPSG:3857'
lambert = 'EPSG:2154'  # to compute accurate distance on mainland France
lat_long = 'EPSG:4326'
Paris_center = Point( 2.3482,48.8534)





rail_stations_filename = Path(r"auxiliary_data/emplacement-des-gares-idf-data-generalisee.csv")

#%%


if __name__ == "__main__":
    geodf = geopandas.read_file(IdFM_path / rail_stations_filename, sep = ";")

    def coord_to_point(coord_str):
        lat_str, long_str = coord_str.split(",")
        return Point((float(long_str), float(lat_str)))

    geodf['geometry'] = geodf['Geo Point'].apply(coord_to_point)

    geodf = geodf.set_crs(lat_long)
    geodf = geodf.to_crs(lambert)


    Paris_center_df = geopandas.GeoDataFrame({'geometry': [Paris_center] * len(geodf)}, crs=lat_long)
    Paris_center_df = Paris_center_df.to_crs(lambert)
    distances_to_center = geodf.distance(Paris_center_df).to_numpy()  # in the lat, long system


    fig, ax = plt.subplots(figsize=(34, 24))
    geodf.plot(ax=ax, markersize=5)
    Paris_center_df.plot(markersize=10, c='r', ax=ax)

    # plt.axis([2.13, 2.54, 48.70, 49.00])   # lat_long
    # plt.axis([2.385e5, 2.833e5, 6.232e6, 6.268e6])   # mercator
    ctx.add_basemap(ax, crs=geodf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    plt.show()

#%%













#%%


def plot_clusters_location(log_r, id_array, color_per_cluster=None, display_legends=False, zoomed_in=False,
                           plot_most_representative=None, id_names=None, sizes=None):
    """
    Parameters
    ----------
    log_r: np array with shape (n_ind, n_days, n_clusters, n_segments)
    id_array: list of strings, corresponds to the "ID_REF_LIGA" in the table , eg
    color_per_cluster: np array, optional
        If omitted, generate random colors
    zoomed_in: bool, optional (defaults to False)
        If True, adjusts the zoom level toshow only the Paris city and close suburbs.
        Otherwise, show the entire map, which might be too large to see Paris.
    display_legends: bool, optional (defaults to False)
    plot_most_representative: positive int or None, optional
        if positive int, highlight the n stations that are the more representative of
        each cluster (the stations that maximiza the likelihood of belonging to every
        cluster).  Defaults to None (highlight nothing)
    id_names: optional, dict with
        keys = id (int)
        values = list of possible names (in the case of bus lines, two directions of the same line
                    have different names)
        When plot_most_representative is not None, giving a value to id_names will plot the names
        of the most represenbtative stations in the legend.
    sizes: int or np array with shape (n_ind,) optional
        the size of each dot. Defaults to 180.


    Returns
    -------
    distances_stations_to_center: np array with shape
    distances_clusters_to_center: couple of np arrays with shape
        The first element is the mean, the second is the standard deviation.

    All distances are in meters
    """

    n_individuals, _, n_clusters, n_segments = log_r.shape
    proba_cluster_per_ind_day = nan_logsumexp(log_r, axis=3, all_nan_return=np.nan)
    log_rho_ik = np.nansum(proba_cluster_per_ind_day, axis=1)   # size: (n_individuals, n_clusters)
    cluster_assign = np.argmax(log_rho_ik, axis=1)

    rail_stations_filename = Path(r"auxiliary_data/emplacement-des-gares-idf-data-generalisee.csv")


    geodf = geopandas.read_file(IdFM_path / rail_stations_filename, sep = ";")

    def coord_to_point(coord_str):
        lat_str, long_str = coord_str.split(",")
        return Point((float(long_str), float(lat_str)))
    geodf['geometry'] = geodf['Geo Point'].apply(coord_to_point)


    geodf = geodf.set_crs(lat_long)
    geodf = geodf.to_crs(lambert)
    geodf.rename(columns={"id_ref_ZdC":"id_ref_lda"}, inplace=True)
    geodf['id_ref_lda'] = geodf['id_ref_lda'].astype(float).astype(int)  # from string to int
    geodf = geodf.drop_duplicates('id_ref_lda')

    geodf = geopandas.GeoDataFrame(sort_dataframe_by(geodf, 'id_ref_lda', id_array))
    if sizes is None: sizes = 180
    if type(sizes) == int: sizes = sizes*np.ones(n_individuals)
        # now sizes is an array with shape (n_individuals,)

    geodf["sizes"] = sizes
    sorted_geodf = geodf
    Paris_center_df = geopandas.GeoDataFrame({'geometry': [Paris_center] * len(sorted_geodf)}, crs=lat_long)
    Paris_center_df = Paris_center_df.to_crs(lambert)
    distances_stations_to_center = sorted_geodf.distance(Paris_center_df).to_numpy()  # in the lat, long system
    mean_distance_cluster_to_center = np.zeros(n_clusters)
    std_distance_cluster_to_center  = np.zeros(n_clusters)


    id_chatelet_les_halles = 73794
    id_chatelet = 474151
    geodf.loc[geodf['id_ref_lda'] == id_chatelet, 'id_ref_lda'] = id_chatelet_les_halles


    sum_found_stations = 0
    fig, ax = plt.subplots(figsize=(36, 30))
    for k in range(n_clusters):
        this_cluster_ids = id_array[cluster_assign == k]
        this_cluster_df = geodf[geodf['id_ref_lda'].isin(this_cluster_ids)].copy().reset_index()
                # the reset_index is here so that there is no misalignment warning when computing distances
        sum_found_stations += len(this_cluster_df)
        label = f"cluster {k+1}" if (plot_most_representative is None) else None

        this_cluster_df.plot(ax=ax, markersize=this_cluster_df['sizes'], color=color_per_cluster[k,:]*0.9, label=label)
            # we lower by 10% to see better the bright colors against the clear background

        Paris_center_df = geopandas.GeoDataFrame({'geometry': [Paris_center] * len(this_cluster_df)}, crs=lat_long).to_crs(lambert)
        mean_distance_cluster_to_center[k] = np.nanmean(this_cluster_df.distance(Paris_center_df, align=False).to_numpy())
        std_distance_cluster_to_center[k]  = np.nanstd( this_cluster_df.distance(Paris_center_df, align=False).to_numpy())




    if plot_most_representative != None:
        log_r_ind_day_clust = nan_logsumexp(log_r, axis=3, all_nan_return=np.nan)  # shape = (n_individuals, n_days, n_clusters)
            # gives the likelihood that each couple (individual, day) belongs to each cluster
        log_r_ind_clust = np.nanmean(log_r_ind_day_clust, axis=1)  # shape = (n_individuals, n_clusters)
            # we take the mean along days to avoid that a station with low number of availabledays is selected
        log_r_ind_clust[np.isnan(log_r_ind_clust)] = -np.inf

        for k in range(n_clusters):
            most_representative_this_cluster = np.argsort(-log_r_ind_clust[:,k])
                # argsort is ascending by default, we want a descending-sorted list
            most_representative_ids = [id_array[id_clust]  for id_clust in most_representative_this_cluster]
                # list of length (n_individuals,) of sorted IDs
            most_representative_selected = most_representative_ids[:plot_most_representative]  # list of length (n_selected_inds,)

            # column title: plot a transparent point
            example_id = most_representative_selected[0]
            selected_geodf = geodf[geodf['id_ref_lda'] == example_id]
            selected_geodf.plot(ax=ax, markersize=0., marker=".", label=f"Cluster {k+1}", alpha=0.)

            for id_selected in most_representative_selected:
                label = id_names[id_selected][0] if display_legends else None
                selected_geodf = geodf[geodf['id_ref_lda'] == id_selected]

                selected_geodf.plot(ax=ax, markersize=60, marker="d", label=label, color=color_per_cluster[k,:]*0.9)
                selected_geodf.plot(ax=ax, markersize=20, marker="$⃝$", color=color_per_cluster[k,:]*0.5)
                # print(selected_geodf)
        if display_legends: plt.legend(title="most representative stations", ncol=n_clusters, loc='upper center')

    else:   # plot_most_representative == None
        if display_legends: plt.legend(ncol=np.ceil(n_clusters/2).astype(int), loc='upper center')




    if zoomed_in: plt.axis([642000, 661000,  6854470, 6869570])   # Lambert
    ctx.add_basemap(ax, crs=geodf.crs.to_string(), source=ctx.providers.Stamen.TonerLite)


    plt.xticks([],[])
    plt.yticks([],[])

    return distances_stations_to_center, (mean_distance_cluster_to_center, std_distance_cluster_to_center)





