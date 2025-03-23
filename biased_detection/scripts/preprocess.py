import pandas as pd
import re
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset/bias_data.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/vectorizer.pkl")

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Preprocess dataset
def preprocess_data(df):
    df['clean_text'] = df['text'].apply(clean_text)
    return df

# Split and vectorize data
def vectorize_data(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['class']  # 1 = biased, 0 = unbiased

    # Ensure models directory exists
    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    
    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved at {VECTORIZER_PATH}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = vectorize_data(df)
    print("Preprocessing complete.")
