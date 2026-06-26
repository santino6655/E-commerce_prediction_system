from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(
    open('purchase_model.pkl', 'rb')
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    browsing = float(request.form['browsing'])
    cart = float(request.form['cart'])
    category = float(request.form['category'])
    duration = float(request.form['duration'])

    features = np.array([[
        age,
        browsing,
        cart,
        category,
        duration
    ]])

    prediction = model.predict(features)

    result = (
        "LIKELY TO PURCHASE"
        if prediction[0] == 1
        else "NOT LIKELY TO PURCHASE"
    )

    return render_template(
        'index.html',
        prediction=result
    )

@app.route('/dashboard')
def dashboard():

    df = pd.read_csv(
        'ecommerce_dataset.csv'
    )

    total_customers = len(df)

    total_purchases = df['purchase'].sum()

    avg_cart = round(
        df['cart_value'].mean(),
        2
    )

    avg_duration = round(
        df['visit_duration'].mean(),
        2
    )

    return render_template(
        'dashboard.html',
        total_customers=total_customers,
        total_purchases=total_purchases,
        avg_cart=avg_cart,
        avg_duration=avg_duration
    )

if __name__ == '__main__':
    app.run(debug=True)