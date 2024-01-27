import dremio_client.lib as dlib
import numpy as np

import pandas as pd
import h3pandas
from datetime import datetime
import time

from pyarrow import flight
from pyarrow.flight import FlightClient
import pyarrow.dataset as ds
import polars as pl

from rapidfuzz import fuzz
import datetime
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from helper import clean_text, apply_rapidfuzz_matching

bakong_data = pd.to_csv('dataset/bakong_data.csv')
aba_data = pd.to_csv('dataset/aba_data.csv')

print("##"*50)
print("Bakong data: ", bakong_data.shape)
print("##"*50)
print("ABA data: ", aba_data.shape)
print("##"*50)

aba_data['AC_NAME'] = aba_data['AC_NAME'].str.upper()
aba_data['AC_NAME'] = aba_data['AC_NAME'].str.strip()
aba_data['AC_NAME'] = aba_data['AC_NAME'].str.replace(" ", "")
aba_data = aba_data.drop_duplicates()
aba_data = aba_data.dropna()

print("##"*50)
print("ABA Data has been cleaned up")
print("##"*50)

bakong_data.drop('Unnamed: 0', axis=1, inplace=True)
bakong_data.head()
bakong_data['Unique_MerchantName'] = bakong_data['Unique_MerchantName'].str.upper()
bakong_data['Unique_MerchantName'] = bakong_data['Unique_MerchantName'].str.strip()
bakong_data['Unique_MerchantName'] = bakong_data['Unique_MerchantName'].str.replace(" ", "")

print("##"*50)
print("Bakong Data has been cleaned up")
print("##"*50)

print("Selecting Sample Tests...")
bk_cust = bakong_data.head(1000)
aba_cust = aba_data.head(10000)

print("##"*50)
print("Data Sample Tests... Here")
print("##"*50)
print(bk_cust.shape)
print(aba_cust.shape)

print("Creating new data frame")
data = pd.DataFrame({"bakong_cust" : bk_cust['Unique_MerchantName'] , "aba_cust" : aba_cust['AC_NAME']})
data.head()

print("##"*50)
print("Cleaning Text data")
print("##"*50)

data['bakong_cust'] = data['bakong_cust'].apply(clean_text)
data['aba_cust'] = data['aba_cust'].apply(clean_text)


print("##"*50)
print("Converting to dask dataframe...")
print("##"*50)

dask_df = dd.from_pandas(data, npartitions=500)
print(dask_df.shape)


print("##"*50)
print("Clean Dask Cluster and Client...")
print("##"*50)
client = Client(n_workers=10, memory_limit="5GB")
print("Cluster info \n", client)


print("##"*50)
print("Creating dataframe to dask data list...")
print("##"*50)

aba_cust_list = dask_df['aba_cust'].compute().tolist()
meta = {'bakong_cust': 'object', 'aba_cust': 'object', 'score': 'float'}

print("##"*50)
print("Start Computing...")
print("##"*50)

result_2 = dask_df.map_partitions(apply_rapidfuzz_matching, aba_cust_list, meta=meta)
start = datetime.datetime.now()
print("Start time:", datetime.datetime.now())
results_df_2 = result_2.compute()
end = datetime.datetime.now()
print("End time:", end)
print("Duration:", end - start)


print("##"*50)
print("Saving Dataset Result to CSV ...")
print("##"*50)

results_df_2.to_csv('dataset/customer_matching_datasets_results.csv')

print("##"*50)
print("Saving Dataset Result to CSV ...")
print("##"*50)