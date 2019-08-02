import pandas as pd
import itertools
import os,sys,inspect
from pandas import DataFrame
from math import sqrt,ceil,floor
import numpy as np
from sklearn.externals import joblib
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import properties as pr

def config_params1(parameter):
    """Function to select parameter from a list of hard coded values """

    p = parameter['p']
    q = parameter['q']
    d = parameter['d']
    m = parameter['m']
    pdq_m = list(itertools.product(p, d, q,m))      #Generate all different combinations of p, q and q triplets
    params = [[(x[0], x[1], x[2]),(x[0], x[1], x[2], x[3])] for x in pdq_m]
    return params

def config_params0(data,parameter):
    """Find THe various configuration of models
     p is calculated from the ACF graph. p is the value on either side 
     of the point at which the ACF line crosses the 0.96 confidence interval
     For q , it is the number of high Correlations before the confidence 
     interval 1.96 
     d and m values are hardcoded"""
    model = []
    #Range of value of p
    acf = sm.graphics.tsa.acf(data.diff().dropna())
    for i in range(len(acf)):
        acf[i] = abs(acf[i]*10)
        if (ceil(acf[i])) <= 2:
            p = range(ceil(acf[i])-1,ceil(acf[i])+2)
            break

    #range of value of q
    pacf = sm.graphics.tsa.pacf(data.diff().dropna())
    for i in range(len(pacf)):
        pacf[i] = abs(pacf[i]*10)
        if (ceil(pacf[i])) <= 2:
            q = range(ceil(pacf[i])-1,ceil(pacf[i])+2)
            break

	# define config lists
    p_params = p
    d_params = parameter['d']
    q_params = q
    m_params = parameter['m']
    #P_params = p
    #D_params = [0, 1]
    #Q_params = q
    
    pdq_m = list(itertools.product(p_params, d_params, q_params,m_params))      #Generate all different combinations of p, q and q triplets
    params = [[(x[0], x[1], x[2]),(x[0], x[1], x[2], x[3])] for x in pdq_m]
    return params

def gridSearch(train,cfg_list):
    results = []
    model = {}
    for index in range(len(cfg_list)):
        order, sorder = cfg_list[index]
	# define model
        temp_dict = {}
        sarima = sm.tsa.statespace.SARIMAX(train,order= order,seasonal_order=sorder,
                                enforce_stationarity=False, enforce_invertibility=False).fit()
        mean_error = MeanError(sarima)
        temp_dict.update({'order':order,'sorder':sorder,'model':sarima,'meanError':mean_error})
        #print("\n {}".format(temp_dict))
        results.append(temp_dict)
    return results

def MeanError(algorithm):
    residuals = DataFrame(algorithm.resid)
    residual_mean = abs(residuals.mean())
    return residual_mean[0]

def SARIMA_forecasting(Data,Time_Column,Target_Column):
    params = pr.param
    min_err_param = {}
    fmodel = {}
    fparam = {}
    #use the below line for finding p and q values from the ACF and PACF graph
    #config = config_params0(Data,params)

    #uncomment the below line for hardcoding the parameters
    config = config_params1(params)

    #print(len(config))
    scores = gridSearch(Data,config)
    min_err_param.update(min(scores, key=lambda x:x['meanError']))
    fmodel.update({'SARIMA Model':min_err_param.pop('model')})
    fparam.update(min_err_param)
    
    joblib.dump(fmodel['SARIMA Model'],pr.path_model)
    return fparam