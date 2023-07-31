from pathlib import Path


IdFM_path = Path(r"/home/hugues_moreau/Documents/datasets/public_transport/ile_de_france_mobilites")

"""
Expected file path :

[IdFM_path]
    data_rf_2015
        2015S1_NB_FER.csv
        2015S2_NB_FER.csv
        ...
    ...
    data_rf_2021

    data_rs_2015
        2015S1_NB_SURFACE.csv
        2015S2_NB_SURFACE.csv
        ...
    ...
    data_rs_2021
        2021S1_NB_SURFACE.csv
        2021S2_NB_SURFACE.csv


    auxiliary_data
        export-paris0.csv
        jours-ouvres-week-end-feries-france-2010-a-2030.csv
        emplacement-des-gares-idf-data-generalisee.csv
        traces-des-lignes-regulieres-de-bus-en-ile-de-france.csv
"""

# The files may be .csv or .txt
# an underscore may be present for some semesters and absent for others
# Also, the year 2021 is divided into trimesters instead of semesters. ("2019_T1_NB_SURFACE.txt")

# All of this does not matter, for the code opens every file which filename does not contain a 'profil' substring

# /!\ data-rf-2020 is a bit particular as it also contains archives of previous years. You will need to put is in the
# data-rf-2020 yourself.

# Links to download the datasets are available here:
# https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-ferre/export/
# https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-surface/export/


# The weather comes from
# https://www.historique-meteo.net/site/export.php?ville_id=188

# And the list of holidays from
# https://ardennemetropole.opendatasoft.com/explore/dataset/jours-ouvresweek-endferies-france-2010-a-2030/


