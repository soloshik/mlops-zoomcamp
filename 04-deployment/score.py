#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[17]:


get_ipython().system('python -V')


# In[27]:


import pickle
import pandas as pd
import numpy as np
import os


# In[19]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[20]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[21]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[22]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[23]:


# Calculate the standard deviation of the predicted duration
std_dev = np.std(y_pred)
print(f"Standard deviation of predicted duration: {std_dev:.2f}")


# Q2

# In[32]:


# Create a dataframe with ride_id and predictions
df_result = pd.DataFrame()
df_result['ride_id'] = '2023/03_' + df.index.astype('str')
df_result['predictions'] = y_pred

# Save the result as parquet
output_file = 'results.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# file size
file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
print(f"Size of the output file: {file_size:.2f}M")


# In[ ]:




