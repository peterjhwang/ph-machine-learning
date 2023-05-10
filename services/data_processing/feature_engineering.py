import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from data_preparation import data_preparation
from utils.aws.s3 import download_from_s3_return_df, upload_str_to_s3


def feature_engineering():
    """
    Download quarterly file from S3, add features, and upload it back to S3
    """
    ## if there is no file, run data_preparation
    try:
        quarterly = download_from_s3_return_df(
            "quarterly.csv", "machine_learning/preprocessing/"
        )
        quarterly["Period"] = pd.to_datetime(quarterly["Period"])
        quarterly.set_index("Period", inplace=True)
    except:
        quarterly = data_preparation()

    original_columns = quarterly.columns

    for i, col in enumerate(original_columns):
        for window_size in [3]:  # range(3,6):
            quarterly[col + "_" + str(window_size) + "_MEAN"] = (
                quarterly[col].rolling(window_size).mean()
            )
            quarterly[col + "_" + str(window_size) + "_STD"] = (
                quarterly[col].rolling(window_size).std()
            )
            quarterly[col + "_" + str(window_size) + "_AAD"] = (
                quarterly[col]
                .rolling(window_size)
                .apply(lambda x: np.mean(np.absolute(x - x.mean())))
            )
            quarterly[col + "_" + str(window_size) + "_MAD"] = (
                quarterly[col]
                .rolling(window_size)
                .apply(lambda x: np.median(np.absolute(x - x.mean())))
            )
            quarterly[col + "_" + str(window_size) + "_MIN"] = (
                quarterly[col].rolling(window_size).min()
            )
            quarterly[col + "_" + str(window_size) + "_MAX"] = (
                quarterly[col].rolling(window_size).max()
            )
            quarterly[col + "_" + str(window_size) + "_MIN_MAX_DIFF"] = (
                quarterly[col + "_" + str(window_size) + "_MAX"]
                - quarterly[col + "_" + str(window_size) + "_MIN"]
            )
            quarterly[col + "_" + str(window_size) + "_MEDIAN"] = (
                quarterly[col].rolling(window_size).median()
            )
            quarterly[col + "_" + str(window_size) + "_IQR"] = (
                quarterly[col]
                .rolling(window_size)
                .apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
            )
            quarterly[col + "_" + str(window_size) + "_ABM"] = (
                quarterly[col]
                .rolling(window_size)
                .apply(lambda x: np.sum(x > x.mean()))
            )
            quarterly[col + "_" + str(window_size) + "_PEAK_COUNT"] = (
                quarterly[col]
                .rolling(window_size)
                .apply(lambda x: len(find_peaks(x)[0]))
            )
            quarterly[col + "_" + str(window_size) + "_SKEWNESS"] = (
                quarterly[col].rolling(window_size).apply(lambda x: stats.skew(x))
            )
            quarterly[col + "_" + str(window_size) + "_KURTOSIS"] = (
                quarterly[col].rolling(window_size).apply(lambda x: stats.kurtosis(x))
            )
            quarterly[col + "_" + str(window_size) + "_ENERGY"] = (
                quarterly[col]
                .rolling(window_size)
                .apply(lambda x: np.sum(x**2 / 100))
            )
            quarterly[col + "_" + str(window_size) + "_SLOPE"] = (
                quarterly[col].rolling(window_size).apply(calculate_slope)
            )

    quarterly.fillna(method="bfill", inplace=True)

    ## upload
    upload_str_to_s3(
        quarterly.to_csv(), "machine_learning/preprocessing/quarterly_feature_added.csv"
    )

    ## log
    application.logger.info("quarterly_feature_added.csv file has been loaded into S3")
    return quarterly
