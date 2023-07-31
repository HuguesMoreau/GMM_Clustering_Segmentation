import pandas as pd
import numpy as np

def sort_dataframe_by(df, by, order):
    """
    Parameters
    ----------
    df: the dataframe to sort
    by: str
        the column whose values we will use to sort
    order: array-like
        the order df values' will follow

    Returns
    -------
    sorted_df: a copy of df (the input, df, has not changed)

    thanks to steboc from
    https://stackoverflow.com/questions/30892355/ordering-dataframe-according-to-a-list-matching-one-of-the-columns-in-pandas
    """

    sorting_df = pd.DataFrame({'sorting_value' : order})
    sorted_df = pd.merge(sorting_df, df, left_on='sorting_value', right_on=by, how='left')
    del sorted_df['sorting_value']
    return sorted_df


if __name__ == '__main__':
    df = pd.DataFrame({"a":[1, 2, 3, 4, 5, 6],
                       "b":[2, 4, 6, 8, 10,12]})
    order      = [5, 3, 1, 2, 4, 6]
    expected_b = [10,6, 2, 4, 8, 12]
    sorted_df = sort_dataframe_by(df, by="a", order=order)

    assert (sorted_df["b"] == np.array(expected_b)).all()



