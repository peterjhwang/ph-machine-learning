from pmdarima import auto_arima
from utils.data_helper.quarterly_helper import find_first_quarter
import pandas as pd
import numpy as np

def predict_current_quarter_value_daily(quarterly, curr_qt, is_mean):
    col_name = quarterly.columns[0]
    x = quarterly[col_name].values
    stepwise_model = auto_arima(x, start_p=1, start_q=1,
                                start_P=0, seasonal=True,
                                d=1, trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    stepwise_model.fit(x)
    next_quarter = find_first_quarter(quarterly.index.max()+pd.DateOffset(days=2)) - pd.DateOffset(days=1)
    total_days = (next_quarter - pd.to_datetime(quarterly.index.max()+pd.DateOffset(days=1))).days
    days_past = len(curr_qt)
    # two predicted values
    # 1.from the current values
    if is_mean:
        current_value = 0 if days_past== 0 else curr_qt['Value'].mean() / days_past * total_days
    else:
        current_value = 0 if days_past== 0 else curr_qt['Value'].sum() / days_past * total_days
    # 2.from the ARIMA result
    arima_result = stepwise_model.predict(n_periods=1)[0]
    # predicted the current quarter value
    quarter_value = current_value * (days_past / total_days) + arima_result * (1 - (days_past/total_days))
    quarterly.loc[next_quarter, col_name] = quarter_value
    return quarterly

def predict_current_quarter_value_monthly(quarterly, curr_qt, is_mean):
    ### TODO: combine with daily function
    col_name = quarterly.columns[0]
    x = quarterly[col_name].values
    stepwise_model = auto_arima(x, start_p=1, start_q=1,
                                start_P=0, seasonal=True,
                                d=1, trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)

    stepwise_model.fit(x)
    next_quarter = find_first_quarter(quarterly.index.max()+pd.DateOffset(days=2)) - pd.DateOffset(days=1)
    total_months = 3
    months_past = len(curr_qt)
    # two predicted values
    # 1.from the current values
    if is_mean:
        current_value = 0 if months_past== 0 else curr_qt['Value'].mean() / months_past * total_months
    else:
        current_value = 0 if months_past== 0 else curr_qt['Value'].sum() / months_past * total_months
    # 2.from the ARIMA result
    arima_result = stepwise_model.predict(n_periods=1)[0]
    # predicted the current quarter value
    quarter_value = current_value * (months_past / total_months) + arima_result * (1 - (months_past/total_months))
    quarterly.loc[next_quarter, col_name] = quarter_value
    return quarterly


## missing pieces with ARIMA
def missing_last_values(df):
    for col_name in df.columns:
        #col_name = daily_df.columns[-1]
        if len(df[df[col_name].isnull()])==0:
            continue
        while np.isnan(df[col_name].values).any():
            x = df.loc[df[col_name].notnull(),col_name].values
            stepwise_model = auto_arima(x, start_p=1, start_q=1,
                                        start_P=0, seasonal=True,
                                        d=1, trace=False,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True)

            stepwise_model.fit(x)
            arima_result = stepwise_model.predict(n_periods=1)[0]
            df.iloc[len(x)][col_name] = arima_result
    return df