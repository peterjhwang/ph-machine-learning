import pandas as pd
import os
from io import StringIO
import boto3

def download_from_s3_return_df(filename, file_location):
    ## get cipher key
    client = boto3.client('s3')
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    ## download files into "temp" folder
    client.download_file('economic-indicators', file_location+filename, 'temp/'+filename)
    ## read from "temp" folder
    with open('temp/'+filename) as f:
        lines = f.read()
    df = pd.read_csv(StringIO(lines))
    ## delete the file
    os.remove('temp/'+filename)
    return df

def upload_str_to_s3(df_str: str, file_location: str):
    client = boto3.client('s3')
    client.put_object(Body=df_str, Bucket='economic-indicators', Key=file_location)


def upload_file_to_s3(file_name, object_name, bucket= 'economic-indicators'):
    client = boto3.client('s3')
    client.upload_file(file_name, bucket, object_name)

def download_file_from_s3(file_name, object_name, bucket= 'economic-indicators'):
    client = boto3.client('s3')
    client.download_file(bucket, bucket, object_name, file_name)