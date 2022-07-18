import pandas as pd
from datetime import datetime as dt

def find_first_quarter(date: str):
    min_date = pd.to_datetime(date)
    year, month, date = min_date.year, min_date.month, min_date.day
    #print(year, month, date)
    if min_date > dt(year, 10, 1):
        return dt(year+1, 1, 1)
    elif min_date > dt(year, 7, 1):
        return dt(year, 10, 1)
    elif min_date > dt(year, 4, 1):
        return dt(year, 7, 1)
    elif min_date > dt(year, 1, 1):
        return dt(year, 4, 1)
    else:
        return dt(year, 1, 1)

def find_last_quarter(date: str):
    max_date = pd.to_datetime(date)
    year, month, date = max_date.year, max_date.month, max_date.day
    #print(year, month, date)
    if max_date < dt(year, 3, 31):
        return dt(year, 1, 1)
    elif max_date < dt(year, 6, 30):
        return dt(year, 4, 1)
    elif max_date < dt(year, 9, 30):
        return dt(year, 7, 1)
    else:
        return dt(year, 10, 1)

def create_quarterly_data(data_frame, col_name, is_mean):
    start_date = find_first_quarter(data_frame.index.min())
    end_date = find_last_quarter(data_frame.index.max())
    #print(start_date, end_date)
    data_frame = data_frame[data_frame.index>=start_date].copy()
    curr_qt = data_frame[data_frame.index>=end_date].copy()
    quar_df = data_frame[data_frame.index<end_date].copy()
    if is_mean:
        quarterly = quar_df.resample('Q').mean()
    else:
        quarterly = quar_df.resample('Q').sum()
    quarterly.columns = [col_name]
    return quarterly, curr_qt