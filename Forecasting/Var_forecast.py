import pandas as pd
from pandas import DataFrame
import os,sys,inspect,pickle
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from math import sqrt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller,acf
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.statespace.varmax import VARMAX

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import properties as pr

def VAR_Model(modeldata):
    model = VAR(modeldata)
    res = {}
    AIC=[]
    for i in range(100):
        result = model.fit(i)
        aic = result.aic
        AIC.append(aic)
        if (aic <= pr.AICvalue_limit) and (aic >= -pr.AICvalue_limit):
            break
    lag_order = i-1

    varmodel = model.fit(lag_order)
    residuals = DataFrame(varmodel.resid)
    rmean = abs(residuals.mean())
    #print("Residual Error = {}".format(rmean[0]))
    res.update({'Residual Mean':rmean,'Lag Order':lag_order})
    return varmodel,res

def VARMAXgridsearch(modeldata,cfg_list):
    results = []
    for index in range(len(cfg_list)):
        order = cfg_list[index]
	# define model
        temp_dict = {}
        varmaxmodel = VARMAX(modeldata, order = order).fit()
        residuals = DataFrame(varmaxmodel.resid)
        mean_error = abs(residuals.mean())
        temp_dict.update({'order':order,'model':varmaxmodel,'meanError':mean_error[0]})
        #print("\n {}".format(temp_dict))
        results.append(temp_dict)
    return results 
    
def config_param(parameter):
    """Find THe various configuration of models p and q for VARMAX"""

    p = parameter['p']
    q = parameter['q']
    pq = list(itertools.product(p,q))      #Generate all different combinations of p, q and q triplets
    params = [(x[0], x[1]) for x in pq]
    return params

def VAR_forecasting(Data,Time_Column,Target_Column):
    #For VAR
    finalVARparam = {}
    fmodel = {}
    VARModel,fresult = VAR_Model(Data)
    fmodel.update({'VAR Model':VARModel})
    finalVARparam.update({'VAR':fresult})

    joblib.dump(fmodel['VAR Model'],pr.path_var)
    
    #For VARMAX
    config = config_param(pr.param)
    scores = VARMAXgridsearch(Data,config)
    min_err_param = min(scores, key=lambda x:x['meanError'])
    fmodel.update({'VARMAX Model':min_err_param.pop('model')})
    finalVARparam.update({'VARMAX':min_err_param})

    joblib.dump(fmodel['VARMAX Model'],pr.path_model)
    
    return finalVARparam