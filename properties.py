import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, make_scorer
import re
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
import sys

#Regression
lim=linear_model
reg={'Lasso':lim.Lasso,'Ridge':lim.Ridge}
prg={'Lasso':{'alpha':[0.1,0.2,0.3,0,1]},'Ridge':{'alpha':[0.1,0.2,0.3,0,1]}}
fea_eng={'corr':0.7,'uniqueness':0.7}

#Forecasting
column_date = 'Timestamp'
path_var = sys.path[0]+'/Models/Forecasting/VAR_Model.pkl'
path_model = sys.path[0]+'/Models/Forecasting/LearnedModel.pkl' #Path for saving VARMAX and SARIMA model
#path_sarima = sys.path[0]+'/Models/Forecasting/SARIMA_Model.pkl'
#path_varmax = sys.path[0]+'/Models/Forecasting/VARMAX_Model.pkl'

#For Seasonality Check
"""The below error is the minimum difference that 2 values can have. 
This is to check if the 2 values are approximately equal """
seasonalCheck_err = 0.1

#For SARIMA and VARMAX
"""Parameters for the SARIMA and VARMAX algorithms
  SARIMA - takes the parameters p,q,d and P,Q,D,m
  VARMAX - takes the parameters p and q"""

param = {
    'p' : range(2,4),
    'q' : range(1,3),
    'd' : range(2),
    'm' : [4,12]    # Change the seasonal frequency here
}

# For VAR
"""This limit is to select the lag order value when AIC is close to 0"""
AICvalue_limit = 0.4

#Classification
classification_list = ['svm','rfc']
algo = {'svm':{'param':{'kernel': ["linear", "poly", "rbf", "sigmoid"],
              'C':[0.01, 0.1, 1, 10, 100, 1000] ,
              'degree': [1, 2, 3, 4, 5, 6]  ,
              'gamma': [0.001, 0.0001]
            },
              'estimator':svm.SVC()
        },
       'rfc':{'param':{'n_estimators': [200, 500],
                            'max_features': ['auto', 'sqrt'],
                            'max_depth' : [4,6,8],
                            'criterion' :['gini', 'entropy']
                            },
                'estimator':RandomForestClassifier()
        }
}


