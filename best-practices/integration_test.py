import os
import pandas as pd
from datetime import datetime
import subprocess
import s3fs

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def main():
    # Step 1: Create the test DataFrame
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(1, 2), dt(1, 10)),
        (None, 2, dt(1, 2), dt(1, 10)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

    df_input = pd.DataFrame(data, columns=columns)

    # Step 2: Set up S3 endpoint and credentials
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'dummy')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'dummy')

    options = {
        'key': AWS_ACCESS_KEY_ID,
        'secret': AWS_SECRET_ACCESS_KEY,
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL,
            'use_ssl': False,
            'verify': False
        }
    }

    # Step 3: Save the test DataFrame to S3
    input_file = 's3://nyc-duration/in/2023-01.parquet'

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    print("Test data uploaded to S3.")

    # **NEW STEP**: Delete existing output file if it exists
    output_file = 's3://nyc-duration/out/2023-01.parquet'
    s3 = s3fs.S3FileSystem(
        key=AWS_ACCESS_KEY_ID,
        secret=AWS_SECRET_ACCESS_KEY,
        client_kwargs={
            'endpoint_url': S3_ENDPOINT_URL,
            'use_ssl': False,
            'verify': False
        }
    )
    if s3.exists(output_file):
        s3.rm(output_file)
        print(f"Deleted existing output file: {output_file}")

    # Step 4: Run the batch.py script
    print("Running batch.py script...")
    return_code = subprocess.call(['python3', 'batch.py', '2023', '1'])
    if return_code != 0:
        print("batch.py script failed.")
        exit(1)

    print("batch.py script completed.")

    # Step 5: Read the output data from S3
    df_result = pd.read_parquet(
        output_file,
        storage_options=options
    )

    # Step 6: Calculate the sum of predicted durations
    total_predicted_duration = df_result['predicted_duration'].sum()
    print(f"Total predicted duration: {total_predicted_duration:.2f}")

if __name__ == "__main__":
    main()
