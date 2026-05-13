from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('ridgeModel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    location = request.form.get('location')
    sqft = request.form.get('sqft')
    bath = request.form.get('bath')
    bhk = request.form.get('bhk')

    input_data = pd.DataFrame(
        [[location, sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    prediction = model.predict(input_data)[0]

    return render_template(
        'index.html',
        prediction_text=f"Predicted Price: ₹ {round(prediction,2)} Lakhs"
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)