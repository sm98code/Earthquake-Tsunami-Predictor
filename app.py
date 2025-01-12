from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and column structure
model = joblib.load('earthquake_tsunami_predictor.pkl')
model_columns = joblib.load('model_columns.pkl')  # Ensure this was saved during training

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data from the form
    data = request.form

    # Extract numeric inputs
    magnitude = float(data['magnitude'])
    cdi = float(data['cdi'])
    mmi = float(data['mmi'])
    tsunami = int(data['tsunami'])
    sig = int(data['sig'])
    nst = int(data['nst'])
    dmin = float(data['dmin'])
    gap = int(data['gap'])
    depth = float(data['depth'])
    latitude = float(data['latitude'])
    longitude = float(data['longitude'])

    # Extract categorical inputs
    alert = data['alert']
    magType = data['magType']

    # Create a base dictionary
    input_data = {
        "magnitude": magnitude,
        "cdi": cdi,
        "mmi": mmi,
        "tsunami": tsunami,
        "sig": sig,
        "nst": nst,
        "dmin": dmin,
        "gap": gap,
        "depth": depth,
        "latitude": latitude,
        "longitude": longitude,
    }

    # One-hot encode 'alert'
    input_data["alert_red"] = 1 if alert == "red" else 0
    input_data["alert_yellow"] = 1 if alert == "yellow" else 0

    # One-hot encode 'magType'
    for mag in ["mb", "md", "ml", "ms", "mw", "mwb", "mwc", "mww"]:
        input_data[f"magType_{mag}"] = 1 if magType == mag else 0

    # Convert to DataFrame and handle missing columns
    input_df = pd.DataFrame([input_data])
    for col in model_columns:
        if col not in input_df:
            input_df[col] = 0  # Fill missing columns with 0

    # Reorder columns to match model
    input_df = input_df[model_columns]

    # Make prediction
    prediction = model.predict(input_df)[0]
    result = "Tsunami" if prediction == 1 else "No Tsunami"

    return jsonify({"Prediction": result})

if __name__ == "__main__":
    app.run(debug=True)