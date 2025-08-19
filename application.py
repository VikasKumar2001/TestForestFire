from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

ridge_model = pickle.load(open('Models/ridecv.pkl', 'rb'))
standard_scaler=pickle.load(open('Models/standard_scaler.pkl', 'rb'))  
 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        input_data = standard_scaler.transform(input_data)
        result = ridge_model.predict(input_data)

        return render_template('input.html', results=result[0])

    else:
        return render_template('input.html')

if __name__=='__main__':
    app.run(debug=True)