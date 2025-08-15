import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
from autots import AutoTS
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import joblib
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')


import streamlit as st
st.title(("Future Forex Currency Price Prediction Model"))

options = {
    'AUSTRALIAN DOLLAR': 'AUS',
    'EURO': 'EUR',
    'NEW ZEALAND DOLLAR': 'NZE',
    'GREAT BRITAIN POUNDS': 'UK',
    'BRAZILIAN REAL': 'BRA',
    'CANADIAN DOLLAR': 'CAN',
    'CHINESE YUAN$': 'CHI',
    'HONG KONG DOLLAR': 'HKO',
    'INDIAN RUPEE': 'IND',
    'KOREAN WON$': 'KOR',
    'MEXICAN PESO': 'MEX',
    'SOUTH AFRICAN RAND$': 'SAF',
    'SINGAPORE DOLLAR': 'SIN',
    'DANISH KRONE': 'DEN',
    'JAPANESE YEN$': 'JAP',
    'MALAYSIAN RINGGIT': 'MAL',
    'NORWEGIAN KRONE': 'NOR',
    'SWEDEN KRONA': 'SWE',
    'SRILANKAN RUPEE': 'SRI',
    'SWISS FRANC': 'SWI',
    'NEW TAIWAN DOLLAR': 'TAI',
    'THAI BAHT': 'THA'
}

model_dict = {'AUSTRALIAN DOLLAR': 'Prophet',
    'EURO': 'Auto',
    'NEW ZEALAND DOLLAR': 'Prophet',
    'GREAT BRITAIN POUNDS': 'Auto',
    'BRAZILIAN REAL': 'Auto',
    'CANADIAN DOLLAR': 'Auto',
    'CHINESE YUAN$': 'Prophet',
    'HONG KONG DOLLAR': 'Auto',
    'INDIAN RUPEE': 'Arima',
    'KOREAN WON$': 'Arima',
    'MEXICAN PESO': 'Auto',
    'SOUTH AFRICAN RAND$': 'Prophet',
    'SINGAPORE DOLLAR': 'Prophet',
    'DANISH KRONE': 'Prophet',
    'JAPANESE YEN$': 'Arima',
    'MALAYSIAN RINGGIT': 'Auto',
    'NORWEGIAN KRONE': 'Prophet',
    'SWEDEN KRONA': 'Prophet',
    'SRILANKAN RUPEE': 'Arima',
    'SWISS FRANC': 'Prophet',
    'NEW TAIWAN DOLLAR': 'Auto',
    'THAI BAHT': 'Prophet'

}

df = pd.read_csv(r"C:\Users\killi\OneDrive\Documents\Other_Stuff\Data_Internship\Project_1\data\Foreign_Exchange_Rates.xls")

#Clean and prep data 
df.drop(df.columns[-1], axis=1, inplace=True)
import datetime
df['Time Serie'] = pd.to_datetime(df['Time Serie'], format="%d-%m-%Y")
for i in range(len(df.columns)-2):
    df[df.columns[i+2]] = pd.to_numeric(df[df.columns[i+2]], errors='coerce')

#Data imputation
for i in range(len(df.columns)-2):
    for j in range(len(df[df.columns[i+2]])):
        if np.isnan(df.loc[j, df.columns[i+2]]):
            if np.isnan(df.loc[j-1, df.columns[i+2]])==False and np.isnan(df.loc[j+1, df.columns[i+2]])==False:
                df.loc[j, df.columns[i+2]] = np.mean([df.loc[j+1, df.columns[i+2]], df.loc[j-1, df.columns[i+2]]])
            elif np.isnan(df.loc[j-1, df.columns[i+2]])==False:
                df.loc[j, df.columns[i+2]] = df.loc[j-1, df.columns[i+2]]
            elif np.isnan(df[df.columns[i+2]][j+1])==False:
                df.loc[j, df.columns[i+2]] = df.loc[j+1, df.columns[i+2]]

data = df.copy()

#function to make predictions, we'll use the code from analysis.ipynb file and make a function which would return forecasts


def make_forecast(selected_option,forecast):
    to_load = options[selected_option]
    model_used = model_dict[selected_option]
    model = joblib.load(r"C:\Users\killi\OneDrive\Documents\Other_Stuff\Data_Internship\Project_1\models\{}".format(to_load))
    if model_used == "Auto":
        prediction = model.predict()
        forecast = prediction.forecast
    elif model_used == "Prophet":
        future_dates = model.make_future_dataframe(periods=forecast, include_history=True, freq="B")
        forecast = model.predict(future_dates)
    elif model_used == "Arima":
        predictions = model.predict(start=len(df["Time Serie"]), end=len(df["Time Serie"])+100)
        forecast = predictions
    return (forecast, model_used, model)



    #currently the model is trained on every submit action from streamlit, find a solution to this problem so that on every submit action, a pretrained model for each currecncy is loaded and inferenced.
    
with st.form(key='user_form'):
    # Add input widgets to the form
    # Create the selectbox
    selected_option = st.selectbox('Choose a currency:', options)
    forecast = st.number_input(
    "Enter an integer",  # Label displayed to the user
    min_value=1,         # Minimum value allowed
    max_value=100,      # Maximum value allowed
    value=1,            # Default value
    step=1              # Increment step
)
    
    submit_button = st.form_submit_button(label='Generate Predictions')

if submit_button:
    
    forecast, used, model = make_forecast(selected_option,forecast)

    if used == "Auto":    
        st.write(forecast)
        st.line_chart(forecast)
        st.dataframe(forecast)
    elif used == "Arima":
        print (forecast)
    elif used == "Prophet":
        fig = model.plot(forecast)
        st.pyplot(fig)
