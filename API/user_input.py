from flask import send_from_directory, abort, Flask, jsonify, abort, request, render_template
import os
#importing classes and Libiaries
import pandas as pd
import numpy as np
import re
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import properties as pr
from Forecasting import SeasonalVerification as f
from Classification import classification_train as c
from Regression import regr as r
#end of importing

app = Flask(__name__)

le = LabelEncoder()

app.config["CSV"]="csv_files"
#F:\pytwrksp\Internship\AutoML\API\csv_files

file_name =""
type =""
target =""
timestamp =""

@app.route('/auto_ml/upload_csv', methods=['POST'])
def receive_csv():
    global file_name
    if request.method=="POST":
        if request.files:
            csv=request.files['csv']
            csv.save(os.path.join(app.config["CSV"],csv.filename))
            file_name=csv.filename
            return jsonify({'Status':'File uploaded'})

@app.route('/auto_ml/metadata', methods=['POST'])
def df_csv():
    global file_name,timestamp,target,type
    data=request.get_json(force=True)
    type=data['type']
    if(data['type']=='forecasting'):
        timestamp=data['timestamp']
    target=data['target']
    return jsonify({'Status':'Data Uploaded Successfully'})


@app.route('/auto_ml/train', methods=['POST'])
def train_data():
    df = pd.read_csv('csv_files/' + file_name)
    if (type == 'regression'):
        rep = r.excecute_regr(df, target)
    if (type == 'forecasting'):
        rep = f.Read_Dataset(df, timestamp, target)
    if(type=='classification'):
        rep=c.Classification(df,target)
    return jsonify(rep)

if __name__ == '__main__':
    app.run(debug=True)