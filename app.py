import pickle

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('heart.pkl', 'rb'))
except FileNotFoundError:
    model = None  # Handle case where model is not available


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
            result = 'Model Not Available'

    except Exception as e:
        result = f'Error: {str(e)}'

    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
