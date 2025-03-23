import joblib
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
from dotenv import load_dotenv
import sys
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("PERSPECTIVE_API_KEY")
URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

if not API_KEY:
    raise ValueError("Error: PERSPECTIVE_API_KEY is missing from .env file. Ensure it is correctly set.")

# Resolve absolute paths for model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/bias_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/vectorizer.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Error: Model or vectorizer file not found. Train the model first.")

# Load ML model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Import `clean_text` from preprocess script if available
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from scripts.preprocess import clean_text
except ImportError:
    raise ImportError("Error: 'clean_text' function not found in scripts/preprocess.py. Ensure the file exists.")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Explicitly allow all origins

# Route to serve the frontend (index.html)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve index.html from /api/templates/

# Function to analyze text using Perspective API
def analyze_perspective(text):
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {
            "TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {}
        }
    }

    try:
        response = requests.post(f"{URL}?key={API_KEY}", json=data)
        response.raise_for_status()
        result = response.json()

        return (
            result["attributeScores"]["TOXICITY"]["summaryScore"]["value"],
            result["attributeScores"]["IDENTITY_ATTACK"]["summaryScore"]["value"],
            result["attributeScores"]["INSULT"]["summaryScore"]["value"]
        )
    except requests.exceptions.RequestException as e:
        return {"error": f"Perspective API Error: {e}"}, 500  # Return error message & HTTP 500

# API route for bias detection
@app.route('/detect', methods=['POST'])
def detect_bias():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # ML Model Prediction
        try:
            clean_text_input = clean_text(text)
            text_vector = vectorizer.transform([clean_text_input])
            ml_prediction_proba = model.predict_proba(text_vector)[0][1]  # Get probability
            ml_prediction = 1 if ml_prediction_proba > 0.63 else 0  # Only flag if high confidence
        except Exception as ml_error:
            return jsonify({"error": f"ML Model Prediction Error: {ml_error}"}), 500

        # Perspective API Prediction
        api_response = analyze_perspective(text)

        if isinstance(api_response, tuple):  # If Perspective API worked
            toxicity, identity_attack, insult = api_response
        else:
            return jsonify(api_response)  # Return error message from analyze_perspective

        # Adjusted Sensitivity
        bias_threshold = 0.63 # Increased threshold to be less sensitive
        low_confidence_threshold = 0.48  # Buffer zone to avoid misclassifications
        
        # Less sensitive final bias classification
        final_bias = 1 if (
            ml_prediction == 1 or
            (toxicity > bias_threshold and toxicity > low_confidence_threshold) or
            (identity_attack > bias_threshold and identity_attack > low_confidence_threshold) or
            (insult > bias_threshold and insult > low_confidence_threshold)
        ) else 0

        return jsonify({
            "text": text,
            "toxicity": toxicity,
            "identity_attack": identity_attack,
            "insult": insult,
            "ml_bias": int(ml_prediction),
            "final_bias": final_bias
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
