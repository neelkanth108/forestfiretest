from flask import Flask, jsonify,request,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_model = pickle.load(open('models/scaler.pkl','rb'))
liner_model = pickle.load(open('models/regressor.pkl','rb'))

app=Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictData',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        temp =float(request.form.get('Temperature'))
        RH =float(request.form.get('RH'))
        Ws =float(request.form.get('Ws'))
        Rain =float(request.form.get('Rain'))
        FFMC =float(request.form.get('FFMC'))
        DMC =float(request.form.get('DMC'))
        ISI =float(request.form.get('ISI'))
        classes =float(request.form.get('Classes'))
        region =float(request.form.get('Region'))
        new_data_scaled =standard_model.transform([[temp,RH,Ws,Rain,FFMC,DMC,ISI,classes,region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html',results= result[0])

    else:
        return render_template('home.html')
    




if __name__ == '__main__':
    app.run(debug=True)