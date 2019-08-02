from flask import Flask, jsonify, request
import os,sys,inspect
import pickle
import pandas as pd
from sklearn.externals import joblib
import json
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import properties as pr

app = Flask(__name__)

type_data = ""

def result_forecast(model,data,type_data):
    if(type_data == 'Range'): 
        start = pd.to_datetime(data['Start'])
        end = pd.to_datetime(data['End'])
        results = model.predict(start = start,end = end,dynamic = True)
    elif(type_data == 'Steps'):
        step = data['Steps']
        results = model.forecast(steps = step)
    elif(type_data == 'Date'):
        date = data['Date']
        date = pd.to_datetime(date)
        results = model.predict(start = date,end = date,dynamic = True)
    return results

@app.route('/auto_ml/forecasting/forecast', methods=['POST'])
def df_csv():
    global type_data
    data =request.get_json(force=True)
    type_data =data['Type']
    if 'Model' in data.keys():
        model_input = data['Model']
        if (model_input == 'VAR'):
            step = data['Steps']
            model = joblib.load(pr.path_var)
            results = model.forecast(model.y,steps = step)
    else:
        model = joblib.load(pr.path_model)
        results = result_forecast(model,data,type_data)
        
    pred = pd.DataFrame(results)
    res = pred.to_json(orient= 'index',date_format= 'iso')
    return res

if __name__ == '__main__':
    app.run(debug=True)