
# ==========================================
# STEP 9.8: Test Trained Model
# ==========================================

import os
import joblib

# Model file location
MODEL_PATH = os.path.join(
    "models",
    "recommendation_engine.pkl"
)

# Check whether model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found!")

print("Loading recommendation model...")

# Load trained model
model = joblib.load(MODEL_PATH)

# Display model information
print("\nModel loaded successfully!")

print("\nAvailable components:")
for key in model.keys():
    print("-", key)

print("\nDataset information:")
print("Total movies:", len(model['movies_full']))
print("Total users:", len(model['user_movie_matrix']))

print("\nModel testing completed!")
