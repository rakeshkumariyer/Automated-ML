#importing classes
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
from AutoML import properties as pr
import pickle as p
import os,sys,inspect
#end of importing
model={}
le = LabelEncoder()
colm_list=[]

def train_model(X,Y,report,n):       #method to calculate the scores of different algorithms
    h=1
    report['Algorithms'] = {}  #generating a report for the user to provide the results of each algorithm
    maxi=-9999999
    for reg_model in pr.reg:
        res=pr.reg[reg_model]()
        grid = GridSearchCV(res, pr.prg[reg_model], cv=2, scoring=None,refit=True)
        grid.fit(X, Y)
        s_score = grid.score(X, Y)
        report['Algorithms'][h] = {}
        report['Algorithms'][h]['Name'] = reg_model
        report['Algorithms'][h]['Score'] = s_score
        report['Algorithms'][h]['Hyperparameters Used'] = grid.best_params_

        h += 1
        if(s_score>maxi):
            maxi=s_score
            model['best_model'] = grid.best_estimator_
    return maxi

def test_columns(df,c_columns,report,n,target):  #method to test the non-numerical columns for unique values
    unwanted=[]
    for c_column in c_columns:
            if(re.search('^[-+]?\d+(\.\d+)?$',str(df[c_column][0]))):
                continue
            else:
                df[[c_column]]=le.fit_transform(df[[c_column]])
                uniqueness=len(df[c_column].unique())
                unique_ratio=uniqueness/n
                if(unique_ratio > pr.fea_eng['uniqueness'] and c_column!=target):   #if uniqueness of values of a column is greater
                    df.drop([c_column], inplace=True,axis=1)   #than a threshold, discard the column
                    unwanted.append(c_column)
    c1_columns = [c for c in c_columns if c not in unwanted]
    return c1_columns


def test_corr(df,c_columns,report,n,target):
    unwanted=[]#testing correlation between the columns
    for i in c_columns:
        for j in c_columns[c_columns.index(i)+1:len(c_columns)]:
            if(j not in c_columns):
                break
            if(i==target or j==target):
                break
            else:
                corr, _ = pearsonr(df[i],df[j])
                if(corr>pr.fea_eng['corr'] or corr<-pr.fea_eng['corr']):    #if correlation between two columns is greater
                    sc1=train_model(df[[i]],df[target],report,n)                             #than a threshold discard one of the columns
                    sc2 = train_model(df[[j]], df[target],report,n)
                    if(sc1<sc2):
                        df.drop([i], inplace=True,axis=1)
                        c_columns.pop(c_columns.index(i))
                        break
                    else:
                        df.drop([j], inplace=True, axis=1)
                        unwanted.append(j)
    c1_columns = [c for c in c_columns if c not in unwanted]
    return c1_columns

def excecute_regr(df,target):
    report = {}
    c_columns=df.columns.values.tolist()
    colm_list = df.columns.values.tolist()
    n=df[c_columns[0]].count()
    print('1:',c_columns)
    c_columns=test_columns(df,c_columns,report,n,target)
    print('2:', c_columns)
    c_columns=test_corr(df,c_columns,report,n,target)
    print('2:', c_columns)
    c_columns.remove(target)
    train_model(df[c_columns],df[target],report,n)
    report['Features used for scoring']=c_columns
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    with open(sys.path[0]+'/models/Regression/features', 'wb') as f:
        p.dump(c_columns, f) 			# Store the best features
    f.close()
    with open(sys.path[0]+'/models/Regression/best_model', 'wb') as f:
        p.dump(model['best_model'], f) 		#Store the Best Model
    f.close()
    return report
