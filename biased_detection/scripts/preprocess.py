import pandas as pd
import re
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Define paths
DATASET_PATH = r"C:\Users\MANBHAV\Downloads\labeled_data.csv"
VECTORIZER_PATH = r"C:\Users\MANBHAV\Documents\vectorizer.pkl"  # Adjust path as needed


# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['tweet', 'class']]  # Keep only relevant columns
    return df

# Text cleaning function
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside brackets
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Preprocess dataset
def preprocess_data(df):
    if 'tweet' not in df.columns or 'class' not in df.columns:
        raise ValueError("Dataset must contain 'tweet' and 'class' columns")
    df['clean_text'] = df['tweet'].apply(clean_text)
    return df

# Split and vectorize data
def vectorize_data(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['class']  # 1 = biased, 0 = unbiased
    
    # Save vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved at {VECTORIZER_PATH}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = vectorize_data(df)
    print("Preprocessing complete.")
