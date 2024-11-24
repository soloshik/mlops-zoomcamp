import pandas as pd
from batch import prepare_data
from datetime import datetime
from pandas.testing import assert_frame_equal

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = [
        'PULocationID', 'DOLocationID', 'tpep_pickup_datetime',
        'tpep_dropoff_datetime'
    ]
    df = pd.DataFrame(data, columns=columns)

    # Ensure datetime columns are of datetime dtype
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Convert categorical columns to nullable integer type
    df['PULocationID'] = df['PULocationID'].astype('Int64')
    df['DOLocationID'] = df['DOLocationID'].astype('Int64')

    categorical = ['PULocationID', 'DOLocationID']

    # Expected data after processing
    expected_data = [
        ('-1', '-1', dt(1, 1), dt(1, 10), 9.0),
        ('1', '1', dt(1, 2), dt(1, 10), 8.0),
    ]
    expected_columns = [
        'PULocationID', 'DOLocationID', 'tpep_pickup_datetime',
        'tpep_dropoff_datetime', 'duration'
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    # Print the number of rows in the expected dataframe
    print(f"Number of rows in expected dataframe: {len(expected_df)}")

    # Ensure data types match
    expected_df['PULocationID'] = expected_df['PULocationID'].astype(str)
    expected_df['DOLocationID'] = expected_df['DOLocationID'].astype(str)
    expected_df['tpep_pickup_datetime'] = pd.to_datetime(
        expected_df['tpep_pickup_datetime']
    )
    expected_df['tpep_dropoff_datetime'] = pd.to_datetime(
        expected_df['tpep_dropoff_datetime']
    )
    expected_df['duration'] = expected_df['duration'].astype(float)

    actual_df = prepare_data(df.copy(), categorical)

    # Reset indexes before comparison
    actual_df = actual_df.reset_index(drop=True)
    expected_df = expected_df.reset_index(drop=True)

    # Print the number of rows in the actual dataframe
    print(f"Number of rows in actual dataframe: {len(actual_df)}")

    # Optionally, assert the number of rows is 2
    assert len(actual_df) == 2, f"Expected 2 rows, got {len(actual_df)} rows"

    # Print the actual and expected DataFrames for comparison
    print("\nActual DataFrame:")
    print(actual_df)
    print("\nExpected DataFrame:")
    print(expected_df)

    # Compare the DataFrames
    assert_frame_equal(actual_df, expected_df, check_dtype=False)
