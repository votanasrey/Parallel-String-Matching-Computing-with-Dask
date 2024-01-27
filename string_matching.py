import numpy as np
import pandas as pd
from datetime import datetime
import time
import datetime
from helper import clean_text
from string_grouper import match_strings, match_most_similar, group_similar_strings, compute_pairwise_similarities, StringGrouper


def main():
    bakong_data = pd.read_csv('dataset/bakong_data.csv')
    aba_data = pd.read_csv('dataset/aba_data.csv')

    print("##"*50)
    print("Bakong data: ", bakong_data.shape)
    print("##"*50)
    print("ABA data: ", aba_data.shape)
    print("##"*50)

    aba_data.drop('Unnamed: 0', axis=1, inplace=True)
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

    print("Final Data Shape: ", data.shape)


    print("##"*50)
    print("Start Computing...")
    print("##"*50)
    start = datetime.datetime.now()

    data['score'] = compute_pairwise_similarities(data['bakong_cust'], data['aba_cust'])

    print(data)

    end = datetime.datetime.now()
    print("End time:", end)
    print("Duration:", end - start)

if __name__ == '__main__':
    main()