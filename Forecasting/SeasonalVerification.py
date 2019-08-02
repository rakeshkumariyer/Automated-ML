import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf

import properties as pr
from Forecasting import Sarima_forecast, Var_forecast

def Seasonality(DataSet,timecolumn,target): 
    Data = DataSet                          # Multivariate Data
    uniData = Data.loc[:,[target]]          # Univariate Data
    lag_acf = acf(uniData.diff().dropna(), nlags=50)
    acf1 = []
    acf2 = []
    for i in range(1,5):
        if i%2 == 0:
            acf2.append(lag_acf[i*12])
        else:
            acf1.append(lag_acf[i*12])
    # Check for seasonality with the (12,36) lag and (24,48) lag  
    if (abs(acf1[0]-acf1[1]) < pr.seasonalCheck_err) and (abs(acf2[0]-acf2[1])<pr.seasonalCheck_err):
        #print("Data is seasonal") 
        Sarima_forecast.SARIMA_forecasting(Data = uniData,Time_Column = timecolumn,Target_Column = target)
    else:
        #print("Data is Not Seasonal")
        Var_forecast.VAR_forecasting(Data=Data, Time_Column= timecolumn,Target_Column=target)

def Read_Dataset(dataset,timecolumn,target):
    """ Read and Modify The DataSet
    dataset variable holds the location and name of the Dataset
    timecolumn holds the TIME column"""

    series = pd.read_csv(dataset)
    series[timecolumn] = pd.to_datetime(series[timecolumn])
    series[pr.column_date] = [d.date() for d in series[timecolumn]]
    series[pr.column_date] = pd.to_datetime(series[pr.column_date])
    data = series.drop(timecolumn, axis=1)
    data = data.drop(pr.column_date,axis =1)
    data = data.set_index(series[pr.column_date])
    #data = data.loc['2010-01-01 00:00:00+00:00':'2011-01-01 00:00:00+00:00']
    weekly = data.resample('D').mean()
    Seasonality(DataSet=weekly,timecolumn=timecolumn,target=target)   