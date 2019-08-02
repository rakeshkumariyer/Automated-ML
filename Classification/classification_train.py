import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import os,sys,inspect
from sklearn.externals import joblib

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import properties as pr

dict_le = {}
def Read_Dataset(dataset,target):
    y= dataset.loc[:, target].values
    #Encoding categorical data
    labelencoder_y = LabelEncoder()
    y= labelencoder_y.fit_transform(y)    #Convert the string names to integers[0,1,2....]
    X=dataset.drop(target,axis =1)
    values_le=labelencoder_y.classes_     
    keys_le = labelencoder_y.transform(labelencoder_y.classes_)
    dict_le = dict(zip(keys_le, values_le))
    joblib.dump(dict_le, sys.path[0]+'/Model/Classification/classifier_dict.pkl')  # Store the mapping of the Classification(String names --> Integers)
    return X,y
    
def Classification(Dataset,Target):
    X,y = Read_Dataset(Dataset,Target)

    best_param={}
    best_models={}
    accuracy_best=[]

    for i in range(len(pr.classification_list)):
        classification = pr.classification_list[i]
        grid= GridSearchCV(estimator = pr.algo[classification]['estimator'], 
                    param_grid = pr.algo[classification]['param'], 
                    cv = 5, 
                    scoring = 'accuracy',
                    refit = True) 
        grid_search = grid.fit(X, y)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        best_param.update({classification:{'Parameters':best_parameters}})
        accuracy_best.append({'Algorithm':classification,'Accuracy':best_accuracy})
        optimized_model=grid.best_estimator_
        best_models.update({classification:{'Model':optimized_model}
                           })
    print(best_param)
    maxacc = {}
    maxacc.update(max(accuracy_best, key=lambda x:x['Accuracy']))
    final_algo=maxacc.pop('Algorithm')
    final_model=best_models[final_algo]['Model']
    joblib.dump(final_model, sys.path[0]+'/Models/Classification/saved_model.pkl')
