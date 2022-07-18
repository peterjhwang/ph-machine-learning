from flask_app import application
import pandas as pd
from datetime import datetime as dt
from pmdarima import auto_arima
from utils.aws.s3 import download_from_s3_return_df, upload_str_to_s3
from utils.data_helper.quarterly_helper import find_first_quarter, find_last_quarter, create_quarterly_data
from utils.data_helper.fill_nan_helper import predict_current_quarter_value_daily, predict_current_quarter_value_monthly, missing_last_values
import numpy as np

CURRENT_QUARTER = dt.today() + pd.tseries.offsets.QuarterEnd(normalize=True)
DATA_START = pd.Timestamp(2019,1,1)

# daily datasets (title, label1, calculation if sum)
DAILY_METRICS = [('Bank bill yields', '', False),
                    # ('Broadband usage', '', False), # too recent
                    ('Daily border crossings - arrivals', 'Total', False),
                    ('Daily border crossings - departures', 'Total', False),
                    ('Electricity grid demand', 'New Zealand', True),
                    #('Gross national happiness', '', True),
                    ('Foreign exchange', 'NZD/USD', True),
                    ('Trade weighted index', '', True),
                    ('Imports by commodity (values)', 'All', False),
                    ('Exports by commodity (values)', 'All', False)
                ]

MONTHLY_METRICS = [
                    ('New Zealand Activity Index (NZAC)', 'New Zealand Activity Index (NZAC)', True),
                    ('New Zealand Activity Index (NZAC)', 'NZAC component - Card transaction spend', True),
                    ('New Zealand Activity Index (NZAC)', 'NZAC component - Electricity grid demand', True),
                    ('New Zealand Activity Index (NZAC)', 'NZAC component - Manufacturing index', True),
                    ('New Zealand Activity Index (NZAC)', 'NZAC component - New jobs posted online', True),
                    ('New Zealand Activity Index (NZAC)', 'NZAC component - Traffic index (heavy)', True),
                    ('New Zealand Activity Index (NZAC)', 'NZAC component - Traffic index (light)', True),
                    ('Card transaction total spend', 'Spend (seasonally adjusted)', False),
                    ('Manufacturing Index', 'Performance of manufacturing (overall index)', True)
                ]

def data_download():
    data_df = download_from_s3_return_df('api-data.csv', 'nz-stats/stats-api/')
    data_df = data_df[(data_df['ResourceID'].isin(['CPTRD2', 'CPTRD4', 'CPTRD1', 'CPTRD5'])) | (data_df['Duration'].notnull())].copy()
    data_df['Period'] = pd.to_datetime(data_df['Period'])
    data_df.fillna('', inplace=True)
    data_df.loc[(data_df['Duration'].isnull()) | (data_df['Duration']==''), 'Duration'] = 'P1D'

    meta_df = download_from_s3_return_df('meta.csv', 'nz-stats/stats-api/')
    meta_df.fillna('', inplace=True)

    temp_df = data_df.merge(meta_df, on='ResourceID')
    df = temp_df.groupby(['Subject', 'ResourceID', 'Title', 'Measure', 'Duration', 'Period', 'GeoUnit', 'Geo', 'Label1', 'Label2', 'Label3', 'Unit', 'Multiplier'])[['Value']].sum().reset_index()
    df.loc[df.Duration=='P1D', 'Duration'] = 1
    df.loc[df.Duration=='P7D', 'Duration'] = 7
    df.loc[df.Duration=='P1M', 'Duration'] = 30
    df.loc[df.Duration=='P3M', 'Duration'] = 90

    df = df[(df['GeoUnit'].isnull()) | (df['GeoUnit']=='')].copy()
    df = df[df['Subject']!='COVID-19'].copy()
    df = df[df['Subject']!='COVID-19 cases'].copy()
    df = df[df['Subject']!='COVID-19 testing'].copy()

    df.loc[(df['Label1'].isnull()) | (df['Label1']==''), 'Label1'] = ''
    df.loc[(df['Label2'].isnull()) | (df['Label2']==''), 'Label2'] = ''
    df.loc[(df['Label3'].isnull()) | (df['Label3']==''), 'Label3'] = ''
    return df

def create_quarterly_data_from_daily_metrics(df):
    daily_df = pd.DataFrame()
    for title, label1, is_true in DAILY_METRICS:
            for label2 in df.loc[(df['Title']==title) & (df['Label1']==label1), 'Label2'].unique():
                for label3 in df.loc[(df['Title']==title) & (df['Label1']==label1) & (df['Label2']==label2), 'Label3'].unique():
                    col_name = ','.join([title, label1, label2, label3])
                    temp = df.loc[(df['Title']==title) & (df['Label1']==label1) & (df['Label2']==label2) & (df['Label3']==label3), ['Period', 'Value']].copy()
                    temp.set_index('Period', inplace=True)
                    quarterly, curr_qt = create_quarterly_data(temp, col_name, is_true)
                    quarterly = predict_current_quarter_value_daily(quarterly, curr_qt, is_true)
                    daily_df = daily_df.join(quarterly, how='outer')
    daily_df = daily_df[daily_df.index>=DATA_START].copy()
    return daily_df

def create_quarterly_data_from_monthly_metrics(df):
    monthly_df = pd.DataFrame()
    for title, label1, is_true in MONTHLY_METRICS:
        for label2 in df.loc[(df['Title']==title) & (df['Label1']==label1), 'Label2'].unique():
            for label3 in df.loc[(df['Title']==title) & (df['Label1']==label1) & (df['Label2']==label2), 'Label3'].unique():
                col_name = ','.join([title, label1, label2, label3])
                temp = df.loc[(df['Title']==title) & (df['Label1']==label1) & (df['Label2']==label2) & (df['Label3']==label3), ['Period', 'Value']].copy()
                temp.set_index('Period', inplace=True)
                quarterly, curr_qt = create_quarterly_data(temp, col_name, is_true)
                quarterly = predict_current_quarter_value_monthly(quarterly, curr_qt, is_true)
                monthly_df = monthly_df.join(quarterly, how='outer')
    monthly_df = monthly_df[monthly_df.index>=DATA_START].copy()
    return monthly_df


def data_preparation():
    ## data download
    df = data_download()

    ## daily metrics
    daily_df = create_quarterly_data_from_daily_metrics(df)
    daily_df = missing_last_values(daily_df)

    ## monthly metrics
    monthly_df = create_quarterly_data_from_monthly_metrics(df)
    monthly_df = missing_last_values(monthly_df)

    prepared_df = daily_df.join(monthly_df, how='outer')

    ## upload
    upload_str_to_s3(prepared_df.to_csv(), 'machine_learning/preprocessing/quarterly.csv')

    ## log
    application.logger.info('quarterly.csv file has been loaded into S3')
    return prepared_df


def load_national_gdp():
    gdp_df = download_from_s3_return_df('gdp.csv', 'nz-stats/webscrapping/')
    gdp_df = gdp_df[gdp_df['Level']=='Total GDP'].copy()
    gdp_df['Period'] = pd.to_datetime(gdp_df['Quarter']) + pd.offsets.QuarterEnd(0)
    gdp_df.set_index('Period', inplace=True)
    gdp_df.rename(columns={'Amount': 'national_gdp'}, inplace=True)
    return gdp_df[['national_gdp']]