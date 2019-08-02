from flask import send_from_directory, abort, Flask, jsonify, abort, request, render_template
import os,sys,inspect
from sklearn.externals import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

app = Flask(__name__)

user_input = ""

@app.route('/auto_ml/Classification/predict', methods=['POST'])
def classifier_input():
    global user_input
    data=request.get_json(force=True)
    user_input = data['Data']
    n_len = len(user_input)
    user_input= np.array(user_input)
    user_input = user_input.reshape(1,n_len)
    
    clfmodel_load = joblib.load(sys.path[0]+'\\Model\\Classification\\saved_model.pkl')
    clfenc_load=joblib.load(sys.path[0]+'\\Model\\Classification\\classifier_dict.pkl')
    y_pred_bin=clfmodel_load.predict(user_input)
    y_pred=clfenc_load[y_pred_bin[0]]

    #labelencoder_y = LabelEncoder()
    #value=labelencoder_y.inverse_transform(y_pred)
    
    return jsonify(str(y_pred))

if __name__ == '__main__':
    app.run(debug=True)