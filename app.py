# app.py
import flask
from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import os # Import os module for path manipulation

# --- Define paths relative to the current script ---
# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath('/Users/usufahmed/Desktop/PREGNANCY_RISK/src'))

MODEL_PATH = os.path.join(BASE_DIR, 'maternal_health_risk_model.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'model_features.joblib')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.joblib')
INDEX_HTML_PATH = os.path.join(BASE_DIR, 'index.html') # Assuming index.html is in the same directory

# --- Load the saved model and related objects ---
loaded_model = None
loaded_features = None
loaded_label_encoder = None

try:
    loaded_model = joblib.load('/Users/usufahmed/Desktop/PREGNANCY_RISK/src/maternal_health_risk_model.joblib')
    loaded_features = joblib.load('/Users/usufahmed/Desktop/PREGNANCY_RISK/src/model_features.joblib')
    loaded_label_encoder = joblib.load('/Users/usufahmed/Desktop/PREGNANCY_RISK/src/label_encoder.joblib')
    print("Model assets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model assets: {e}")
    print("Please ensure 'maternal_health_risk_model.joblib', 'model_features.joblib', and 'label_encoder.joblib' are in the same directory as app.py.")
    # In a production app, you might want to raise an exception or log and exit gracefully.
    # For now, we'll let the app run but prediction will fail if assets are truly missing.
except Exception as e:
    print(f"An unexpected error occurred while loading model assets: {e}")

# --- Initialize Flask App ---
# Tell Flask to look for templates in the same directory as app.py
app = Flask(__name__, template_folder=BASE_DIR)

# --- Define Routes ---

@app.route('/')
def home():
    # Render the index.html template. Flask will look for 'index.html'
    # in the 'template_folder' specified above (which is BASE_DIR).
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_features is None or loaded_label_encoder is None:
        return render_template('index.html', prediction_text="Error: Model assets not loaded. Cannot make predictions.")

    if request.method == 'POST':
        form_data = request.form
        new_data_point = {}
        errors = []

        # Validate and collect form data for each feature
        for feature in loaded_features:
            value_str = form_data.get(feature) # Use .get() to avoid KeyError if a field is missing
            if value_str is None or value_str == '':
                errors.append(f"Missing input for '{feature}'.")
                continue
            try:
                # Convert to float, then to appropriate type if needed by model (e.g., int for binary flags)
                # For simplicity, we'll keep everything as float as Random Forest handles it.
                value = float(value_str)
                new_data_point[feature] = value
            except ValueError:
                errors.append(f"Invalid input for '{feature}'. Please enter a numeric value.")

        if errors:
            return render_template('index.html', prediction_text="<br>".join(errors))

        # Check if all features were collected
        if len(new_data_point) != len(loaded_features):
            return render_template('index.html', prediction_text="Error: Not all required features were provided or were invalid.")

        # Convert the dictionary to a pandas DataFrame
        # Ensure the columns are in the correct order as loaded_features
        new_data_df = pd.DataFrame([new_data_point], columns=loaded_features)

        # Make a prediction
        try:
            prediction_encoded = loaded_model.predict(new_data_df)
            prediction_label = loaded_label_encoder.inverse_transform(prediction_encoded)

            # Optionally, get prediction probabilities
            prediction_proba = loaded_model.predict_proba(new_data_df)
            # Find the probability for the predicted class
            predicted_class_index = loaded_label_encoder.transform([prediction_label[0]])[0]
            confidence = prediction_proba[0][predicted_class_index]

            result_text = f"Predicted Maternal Health Risk Level: <b>{prediction_label[0]}</b> (Confidence: {confidence:.2f})"
            return render_template('index.html', prediction_text=result_text)

        except Exception as e:
            # Log the full traceback for debugging in your server's console
            import traceback
            traceback.print_exc()
            return render_template('index.html', prediction_text=f"An unexpected error occurred during prediction. Please check server logs. Error: {e}")

# --- Run the Flask App ---
if __name__ == "__main__":
    # Run Flask in debug mode for development. Set debug=False for production.
    app.run(debug=True,threaded=True)