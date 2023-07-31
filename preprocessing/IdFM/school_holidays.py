"""
School holidays in the Paris region (some school holidays have different dates or each French region)
Source:
    https://vacances-scolaires.education/academie-paris/annee-2014-2015.php
    https://vacances-scolaires.education/academie-paris/annee-2015-2016.php
    etc.
"""

import datetime

def date_dmy(d, m, y):
    # merely changes the format
    return datetime.date(y, m, d)


# start and end are included
start_end_school_holidays = [(date_dmy(20,12,2014),date_dmy( 4, 1,2015)),    # Christmas
                                     # 2015
                             (date_dmy(14, 2,2015),date_dmy( 1, 3,2015)),    # winter holidays
                             (date_dmy(18, 4,2015),date_dmy( 3, 5,2015)),    # spring holidays
                             (date_dmy( 4, 7,2015),date_dmy(31, 8,2015)),    # summer holidays
                             (date_dmy(17,10,2015),date_dmy( 1,11,2015)),    # Haloween
                             (date_dmy(19,12,2015),date_dmy( 3, 1,2016)),    # Christmas
                                     # 2016
                             (date_dmy(20, 2,2016),date_dmy( 6, 3,2016)),    # winter holidays
                             (date_dmy(16, 4,2016),date_dmy( 1, 5,2016)),    # spring holidays
                             (date_dmy( 5, 7,2016),date_dmy(31, 8,2016)),    # summer holidays
                             (date_dmy(19,10,2016),date_dmy( 2,11,2016)),    # Haloween
                             (date_dmy(17,12,2016),date_dmy( 1, 1,2017)),    # Christmas
                                     # 2017
                             (date_dmy( 4, 2,2017),date_dmy(19, 2,2017)),    # winter holidays
                             (date_dmy( 1, 4,2017),date_dmy(17, 4,2017)),    # spring holidays
                             (date_dmy( 8, 7,2017),date_dmy( 3, 9,2017)),    # summer holidays
                             (date_dmy(21,10,2017),date_dmy( 5,11,2017)),    # Haloween
                             (date_dmy(23,12,2017),date_dmy( 7, 1,2018)),    # Christmas
                                     # 2018
                             (date_dmy(17, 2,2018),date_dmy( 4, 3,2018)),    # winter holidays
                             (date_dmy(14, 4,2018),date_dmy(29, 4,2018)),    # spring holidays
                             (date_dmy( 7, 7,2018),date_dmy( 2, 9,2018)),    # summer holidays
                             (date_dmy(20,10,2018),date_dmy( 4,11,2018)),    # Haloween
                             (date_dmy(22,12,2018),date_dmy( 6, 1,2019)),    # Christmas
                                     # 2019
                             (date_dmy(23, 2,2019),date_dmy(10, 3,2019)),    # winter holidays
                             (date_dmy(20, 4,2019),date_dmy( 5, 5,2019)),    # spring holidays
                             (date_dmy( 6, 7,2019),date_dmy( 1, 9,2019)),    # summer holidays
                             (date_dmy(19,10,2019),date_dmy( 3,11,2019)),    # Haloween
                             (date_dmy(21,12,2019),date_dmy( 5, 1,2020)),    # Christmas
                                     # 2020
                             (date_dmy( 8, 2,2020),date_dmy(23, 2,2020)),    # winter holidays
                             (date_dmy( 4, 4,2020),date_dmy(19, 4,2020)),    # spring holidays
                             (date_dmy( 4, 7,2020),date_dmy(31, 8,2020)),    # summer holidays
                             (date_dmy(17,10,2020),date_dmy( 1,11,2020)),    # Haloween
                             (date_dmy(19,12,2020),date_dmy( 3, 1,2021)),    # Christmas
                                     # 2021
                             (date_dmy(13, 2,2021),date_dmy(28, 2,2021)),    # winter holidays
                             (date_dmy(10, 4,2021),date_dmy(25, 4,2021)),    # spring holidays
                             (date_dmy( 6, 7,2021),date_dmy( 1, 9,2021)),    # summer holidays
                             (date_dmy(23,10,2021),date_dmy( 7,11,2021)),    # Haloween
                             (date_dmy(18,12,2021),date_dmy( 2, 1,2022)),    # Christmas
                                     # 2022
                             (date_dmy(19, 2,2022),date_dmy( 6, 3,2022)),    # winter holidays
                             (date_dmy(23, 4,2022),date_dmy( 8, 5,2022)),    # spring holidays
                             (date_dmy( 7, 7,2022),date_dmy(31, 8,2022)),    # summer holidays
                             (date_dmy(22,10,2022),date_dmy( 6,11,2022)),    # Haloween
                             (date_dmy(17,12,2022),date_dmy( 3, 1,2023)),    # Christmas


                            ]