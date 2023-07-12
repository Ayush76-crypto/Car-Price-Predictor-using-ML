from flask import Flask, render_template, url_for, request, redirect

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

model=pickle.load(open('LinearRegressionModel2.pkl','rb'))

car = pd.read_csv("Cleaned car data.csv")

@app.route('/')
def index():
    
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])

def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')
    
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)
    

if __name__ == '__main__':
    app.run(debug=True)