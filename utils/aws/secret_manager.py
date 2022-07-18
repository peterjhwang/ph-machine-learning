import boto3

def get_secret(secret_name, region_name="ap-southeast-2"):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']
