import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load preprocessed data
from preprocess import load_data, preprocess_data, vectorize_data

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "../dataset/bias_data.csv")
models_dir = os.path.join(current_dir, "../models")
model_path = os.path.join(models_dir, "bias_model.pkl")

# Ensure dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

# Load and preprocess dataset
df = load_data(dataset_path)
df = preprocess_data(df)

# Ensure 'class' column exists (based on your CSV structure)
if "class" not in df.columns:
    raise KeyError("Expected column 'class' not found in dataset")

# Vectorize and split data
X_train, X_test, y_train, y_test = vectorize_data(df)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Ensure models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save trained model
joblib.dump(model, model_path)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
