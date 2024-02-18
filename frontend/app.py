import pandas
import streamlit
import pickle
import xgboost
import os
import requests
import json

streamlit.title("Credit Card Fraud Detection App")


streamlit.write("""
         Credit card fraud is a type of identity theft that happens when someone uses your credit card information, **without your permission**, to make unwanted purchases.
         They don't need your physical card, just the details like the number, expiration date, and security code (CCV).
         
         **The web app utilizes machine learning to detect fraudulent credit card transactions based on several criteria:
         time of the transaction (whether it occured on a weekend, or at night time), number of transactions & average amount over a number of days, etc.**
         
         Full code is available on [GitHub.](https://github.com/tiqfds/credit-fraud-detection)
         
         """)

def run():
    
    quarter = streamlit.sidebar.selectbox('Which quarter do you want to analyze?', (1, 2, 3, 4))
    if quarter == 1:
        q_data = '../data/processed/web_app_data/Q1.feather'
    if quarter == 2:
        q_data = '../data/processed/web_app_data/Q2.feather'
    if quarter == 3:
        q_data = '../data/processed/web_app_data/Q3.feather'
    if quarter == 4:
        q_data = '../data/processed/web_app_data/Q4.feather'

    @streamlit.cache_data
    def load_data(q_data):
        df = pandas.read_feather(q_data)
        return df
    
    df = load_data(q_data)

    if streamlit.sidebar.checkbox('Show information about the data'):
        streamlit.write(df.head(10))
        streamlit.write("{0} transactions loaded as training data, containing {1} fraudulent transactions".format(len(df),df.TX_FRAUD.sum()))


if __name__ == '__main__':
    run()