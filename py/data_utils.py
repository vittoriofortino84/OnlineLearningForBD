import pandas as pd

from consts import EVENT_STR, TIME_STR


def merge_x_y(x, y):
    events = pd.Series(y[:][EVENT_STR], name=EVENT_STR)
    times = pd.Series(y[:][TIME_STR], name=TIME_STR)
    res = pd.concat([x, events, times], axis=1)
    return res

