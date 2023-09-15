from flask import Flask,render_template
from flask import request
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score


model = pickle.load(open('Court_model_final.pkl','rb'))


app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
  State =  request.form.get('State')
  Type =  request.form.get('Type')
  Judge_Label =  request.form.get('Judge_Label')
  result = model.predict(np.array([State,Type,Judge_Label]).reshape(1,3))
  result = result.astype('int')
  return str(result)


if __name__ =='__main__':
  app.run(host ='0.0.0.0',port = 80)