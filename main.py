import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from os import environ
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.preprocessing import PolynomialFeatures

def load_model():
    file_path = "../ml-workflow-model-layer/data/model.sav"
    return pickle.load(open(file_path, "rb"))

def polynomialize(input):
    transformer = PolynomialFeatures(degree=10)
    input = input.values.reshape(-1, 1)
    input = transformer.fit_transform(input)
    return input

port = int(environ.get("PORT", 2000))
app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/predict')
def predict():
    time = datetime.now().hour

    model = load_model()
    input = polynomialize(pd.Series(np.array(time)))
    prediction = model.predict(input)

    return jsonify(dict(prediction=round(prediction[0], 2)))

app.run(debug=False, host='0.0.0.0', port=port)
