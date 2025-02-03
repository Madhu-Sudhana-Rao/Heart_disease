import pickle
import os
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), "heart.pkl")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
else:
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/Predict', methods=['POST'])
def predict():
    try:
        features = [
            request.form.get('age', type=int),
            1 if request.form.get('sex') == 'male' else 0,
            request.form.get('cp', type=int),
            request.form.get('trestbps', type=int),
            request.form.get('chol', type=int),
            request.form.get('fbs', type=int),
            request.form.get('restecg', type=int),
            request.form.get('thalach', type=int),
            request.form.get('exang', type=int),
            request.form.get('oldpeak', type=float),
            request.form.get('slope', type=int),
            request.form.get('ca', type=int),
            request.form.get('thal', type=int)
        ]
        input_data = np.array(features).reshape(1, -1)

        if model:
            prediction = model.predict(input_data)[0]
            result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        else:
            result = 'Model Not Available. Please upload the trained model.'

    except Exception as e:
        result = f'Error: {str(e)}'

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
