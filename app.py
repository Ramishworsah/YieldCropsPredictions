from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
import sklearn

print(sklearn.__version__)

# load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # get form values
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # create DataFrame for preprocessing
        features = pd.DataFrame([{
            'Year': Year,
            'average_rain_fall_mm_per_year': average_rain_fall_mm_per_year,
            'pesticides_tonnes': pesticides_tonnes,
            'avg_temp': avg_temp,
            'Area': Area,
            'Item': Item
        }])

        # transform using preprocessor
        transformed_features = preprocessor.transform(features)

        # make prediction
        prediction = dtr.predict(transformed_features)[0]

        return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
