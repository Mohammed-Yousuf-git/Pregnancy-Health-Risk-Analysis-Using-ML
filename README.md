# Maternal Health Risk Prediction

This project predicts the **maternal health risk** during pregnancy (High or Low) using machine learning. The app features an intuitive web interface for instant predictions based on patient health metrics.

---

## 🌟 Features

- **Data Preprocessing**: Handles missing values (imputation with median) and encodes categorical target variables.
- **Predictive Modeling**: Trains a `RandomForestClassifier` to predict risk levels.
- **Risk Factor Analysis**: Finds and visualizes key features influencing risk using feature importances.
- **Patient Profiling**: Shows average metrics for High vs. Low risk groups.
- **Web Application**: User-friendly Flask interface for real-time predictions.
- **Model Persistence**: Saves trained model, feature list, and label encoder for deployment.

---

## 📊 Dataset

**File**: `Dataset - Updated.csv`

**Columns**:

- Age: Age of the pregnant individual.
- Systolic BP: Systolic Blood Pressure.
- Diastolic: Diastolic Blood Pressure.
- BS: Blood Sugar level.
- Body Temp: Body Temperature.
- BMI: Body Mass Index.
- Previous Complications: 1 if yes, 0 if no.
- Preexisting Diabetes: 1 if yes, 0 if no.
- Gestational Diabetes: 1 if yes, 0 if no.
- Mental Health: 1 if yes, 0 if no.
- Heart Rate: Heart Rate.
- Risk Level: Target variable (`High`, `Low`).

---

## 🚀 Prerequisites

- Python 3.x
- pip

---

## 📦 Project Structure

```
. ├── Dataset - Updated.csv
├── app.py
├── train_and_save_model.py  # Script to train and save the model assets
├── index.html
├── maternal_health_risk_model.joblib
 ├── model_features.joblib
└── label_encoder.joblib
```

> **Note:** `.joblib` files are generated after running `train_and_save_model.py`

---

## 🛠️ Setup and Installation

1. **Clone or Download the Project**  
   Download all files to a single directory.

2. **Install Dependencies**  
   Open terminal, navigate to the project directory, and run:
```
!pip install pandas scikit-learn matplotlib seaborn joblib Flask

```

3. **Generate Model Assets**  
- Ensure `Dataset - Updated.csv` is in the same directory as `train_and_save_model.py`.
- In terminal:
  ```
  python train_and_save_model.py
  ```
- This will:
  - Load and preprocess the data
  - Train the RandomForestClassifier
  - Save: `maternal_health_risk_model.joblib`, `model_features.joblib`, `label_encoder.joblib`

4. **Run the Web Application**  
- Ensure `app.py`, `index.html`, and all `.joblib` files are in the same directory.
- Start Flask:
  ```
  python app.py
  ```
- You should see:
  ```
  * Serving Flask app 'app'
  * Debug mode: on
  WARNING: This is a development server. Do not use it in a production deployment.
  Use a production WSGI server instead.
  * Running on http://127.0.0.1:5000
  Press CTRL+C to quit
  ```

---

## 🌐 Usage

1. Open your browser and go to the address shown in your terminal (e.g., `http://127.0.0.1:5000/`).
2. Fill in all patient health metric fields in the form.
3. Click **"Predict Risk"**.
4. The app will display the predicted Risk Level (High/Low) and the confidence score.

---

## 🧠 Model Details

- **Algorithm:** RandomForestClassifier
- **Performance:** 100% accuracy on provided dataset during testing.
- **Top Risk Factors (Feature Importances):**
- Blood Sugar (BS)
- Preexisting Diabetes
- Heart Rate
- BMI
- Gestational Diabetes

---

## 💡 Future Enhancements

- Enhanced UI/UX for the web interface
- Robust input validation on client/server
- Detailed error logging
- Database integration for user or prediction history
- Deploy to cloud platforms (Google Cloud Run, Heroku, AWS)
- Model monitoring (detect drift, track performance)
- Explainability tools (SHAP, LIME support)

---

## 🤝 Contributing

Feel free to fork this repository, open issues, and submit pull requests.

---

## 📄 License

Open-source under the MIT License.

## Contact

Created by Mohammed Yousuf -feel free to contact via yousufmohammed9148@gmail.com or open an issue.
HAPPY CODING
