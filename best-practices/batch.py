import os
import sys
import pickle
import pandas as pd
from datetime import datetime
import s3fs

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Use nullable integer type 'Int64' for proper handling of NaN
    df[categorical] = df[categorical].fillna(-1).astype('Int64').astype('str')

    return df

def read_data(filename, categorical):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if filename.startswith('s3://'):
        if S3_ENDPOINT_URL:
            # Create an S3 filesystem with the custom endpoint and SSL disabled
            s3_fs = s3fs.S3FileSystem(
                key=os.getenv('AWS_ACCESS_KEY_ID', 'dummy'),
                secret=os.getenv('AWS_SECRET_ACCESS_KEY', 'dummy'),
                client_kwargs={
                    'endpoint_url': S3_ENDPOINT_URL,
                    'use_ssl': False,
                    'verify': False
                }
            )
            df = pd.read_parquet(
                filename,
                filesystem=s3_fs
            )
        else:
            df = pd.read_parquet(filename)
    else:
        # For local files
        df = pd.read_parquet(filename)

    df = prepare_data(df, categorical)
    return df

def save_data(df, filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if filename.startswith('s3://'):
        if S3_ENDPOINT_URL:
            # Create an S3 filesystem with the custom endpoint and SSL disabled
            s3_fs = s3fs.S3FileSystem(
                key=os.getenv('AWS_ACCESS_KEY_ID', 'dummy'),
                secret=os.getenv('AWS_SECRET_ACCESS_KEY', 'dummy'),
                client_kwargs={
                    'endpoint_url': S3_ENDPOINT_URL,
                    'use_ssl': False,
                    'verify': False
                }
            )
            df.to_parquet(
                filename,
                engine='pyarrow',
                index=False,
                filesystem=s3_fs
            )
        else:
            df.to_parquet(
                filename,
                engine='pyarrow',
                index=False
            )
    else:
        df.to_parquet(
            filename,
            engine='pyarrow',
            index=False
        )

def get_source_data_url(year, month):
    return f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

def get_input_path(year, month):
    default_input_pattern = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Set up s3_fs
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if input_file.startswith('s3://') and S3_ENDPOINT_URL:
        # Create an S3 filesystem with the custom endpoint and SSL disabled
        s3_fs = s3fs.S3FileSystem(
            key=os.getenv('AWS_ACCESS_KEY_ID', 'dummy'),
            secret=os.getenv('AWS_SECRET_ACCESS_KEY', 'dummy'),
            client_kwargs={
                'endpoint_url': S3_ENDPOINT_URL,
                'use_ssl': False,
                'verify': False
            }
        )
    else:
        s3_fs = None

    # Check if output file already exists
    if s3_fs and s3_fs.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping processing.")
        return  # Exit the script

    # Check if input file exists in S3
    if s3_fs and not s3_fs.exists(input_file):
        # Download from public source
        source_url = get_source_data_url(year, month)
        print(f"Downloading data from {source_url}")
        df_source = pd.read_parquet(source_url)
        # Save to S3
        save_data(df_source, input_file)
        print(f"Uploaded data to {input_file} in S3.")

    # Read data from input file
    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Load the model
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    # Prepare features and make predictions
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    # Prepare the result DataFrame
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    # Save the result DataFrame
    save_data(df_result, output_file)
    print(f"Uploaded result to {output_file} in S3.")

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
