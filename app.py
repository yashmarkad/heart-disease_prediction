from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("E:\ml_heart_dieses\heart_disease_model.pkl")  # Update with your model path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    
    # Create a feature array
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    final_features = [np.array(features)]

    # Make prediction
    prediction = model.predict(final_features)
    output = prediction[0]

    # Display prediction result
    if output == 0:
        return render_template('index.html', prediction_text='The person does not have a heart disease')
    else:
        return render_template('index.html', prediction_text='The person has heart disease')

if __name__ == '__main__':
    app.run(debug=True)
    