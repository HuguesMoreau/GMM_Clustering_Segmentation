"""
This file is only here to provide the list of strikes in the RATP
company between January 2015 and early 2023.
We could have created a .csv file separately, but we decided against it
because, contrary to other data sources, the authors had to manually
compile data from the site https://www.cestlagreve.fr , thus being more
prone to errors than data from another source.
"""

import datetime

def date_dmy(d, m, y):
    # merely changes the format
    return datetime.date(y, m, d)



start_end_strikes = [(date_dmy(31, 1,2023),date_dmy(31, 1,2023)),
                    (date_dmy(19, 1,2023),date_dmy(19, 1,2023)),
                    (date_dmy(13, 1,2023),date_dmy(13, 1,2023)),
                    (date_dmy(10,10,2022),date_dmy(10,10,2022)),
                    (date_dmy( 3, 6,2022),date_dmy( 3, 6,2022)),
                    (date_dmy(23, 5,2022),date_dmy(25, 5,2022)),
                    (date_dmy(25, 3,2022),date_dmy(25, 3,2022)),
                    (date_dmy(18, 2,2022),date_dmy(18, 2,2022)),
                    (date_dmy(15, 2,2021),date_dmy(15, 2,2021)),
                    (date_dmy(17,12,2020),date_dmy(17,12,2020)),
                    (date_dmy(18, 5,2020),date_dmy(18, 5,2020)),
                    (date_dmy(17, 2,2020),date_dmy(17, 2,2020)),
                    (date_dmy( 5,12,2019),date_dmy(13, 1,2020)),
                    (date_dmy(13, 9,2019),date_dmy(13, 9,2019)),
                    (date_dmy( 5, 2,2019),date_dmy( 5, 2,2019)),
                    (date_dmy(22, 3,2018),date_dmy(22, 3,2018)),
                    (date_dmy( 2, 6,2016),date_dmy(20, 6,2016)),
                    (date_dmy(31, 3,2016),date_dmy( 1, 4,2016)),
                    (date_dmy( 9, 3,2016),date_dmy( 9, 3,2016)),
                    (date_dmy(10,12,2015),date_dmy(11,12,2015)),
                    (date_dmy(18,11,2015),date_dmy(18,11,2015)),
                    (date_dmy(15,10,2015),date_dmy(15,10,2015)),
                    (date_dmy(23,10,2015),date_dmy(23,10,2015)),
                    (date_dmy(17, 6,2015),date_dmy(20, 6,2015))]
    # note that both starting day and end day are included






