import pandas
import streamlit as st
import pickle
import xgboost
import os
import requests
import json

st.title("Credit Card Fraud Detection App")


st.write("""
         Credit card fraud is a type of identity theft that happens when someone uses your credit card information, **without your permission**, to make unwanted purchases.
         They don't need your physical card, just the details like the number, expiration date, and security code (CCV).
         
         **The web app utilizes machine learning to detect fraudulent credit card transactions based on several criteria:
         time of the transaction (whether it occured on a weekend, or at nighttime), number of transactions & average amount over a number of days, etc.**
         
         Full code is available on [GitHub.](https://github.com/tiqfds/credit-fraud-detection)
         
         """)
