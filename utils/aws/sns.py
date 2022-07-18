import os
import boto3
from dotenv import load_dotenv
load_dotenv()

SNS_ARN = os.getenv('SNS_ARN')

# SNS
def send_message(typ, message):
    client = boto3.client('sns', region_name = 'ap-southeast-2')
    return client.publish(TopicArn = SNS_ARN,
                   Message = message,
                   Subject = f"{typ} - ph-data-pipeline")