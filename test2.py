
from flask import Flask, jsonify, request
import joblib
import pandas as pd


app = Flask(__name__)

@app.route('/',methods=['POST'])

def predict():
    
    lr = joblib.load("model.pkl")
    json_= request.json
    print(json_,type(json_))
    x_test=pd.DataFrame.from_dict(json_)
    
    
    if lr:
    	preds = lr.predict(x_test)
    	return jsonify({'prediction': str(preds)})


app.run(host='0.0.0.0', port=5000)