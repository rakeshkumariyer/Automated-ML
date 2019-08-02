from flask import send_from_directory, abort, Flask, jsonify, abort, request, render_template
import os
#importing classes
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, make_scorer
import re
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from AutoML import properties as pr
from AutoML.Regression import regr as r
from AutoML.Forecasting import SeasonalVerification as f
from AutoML.Classification import classification_train as c
import pickle as p
import os,sys,inspect
#end of importing

app = Flask(__name__)

le = LabelEncoder()


@app.route('/auto_ml/regression/predict', methods=['POST'])
def df_csv():
    global file_name,timestamp,target,type
    fea=[]
    data=request.get_json(force=True)
    #type=data['type']
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    with open(sys.path[0]+'\models\Regression\\features', 'rb') as f:
        cols = p.load(f)
    f.close()
    for i in cols:
        fea.append(float(data[i]))
    with open(sys.path[0]+'\models\Regression\\best_model', 'rb') as f:
        reg_model = p.load(f)
    f.close()
    val=reg_model.predict([fea])
    print(val)
    #return 'done'
    return jsonify({'Predicted value of the target':val[0]})




if __name__ == '__main__':
    app.run(debug=True)








