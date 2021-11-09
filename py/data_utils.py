import pandas as pd

from consts import EVENT_STR, TIME_STR


def merge_x_y(x, y):
    if len(x) != len(y):
        raise ValueError("x and y have different lengths\n" + str(x) + "\n" + str(y))
    res = pd.concat([x.reset_index(drop=True), y.loc[:, [EVENT_STR, TIME_STR]].reset_index(drop=True)], axis=1)
    return res

