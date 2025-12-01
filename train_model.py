import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate synthetic lifestyle dataset
np.random.seed(42)
samples = 500

# Generate synthetic lifestyle dataset
np.random.seed(42)
samples = 500

data = pd.DataFrame({
    "sleep_hours": np.random.uniform(4, 10, samples),
    "water_glasses": np.random.randint(2, 12, samples),
    "screen_time": np.random.uniform(2, 12, samples),
    "steps": np.random.randint(1000, 15000, samples),
    "mood": np.random.randint(1, 6, samples),  # 1:bad → 5:great
})

# Stress label generation formula
data["stress_level"] = (
    (data["sleep_hours"] < 6).astype(int)
    + (data["water_glasses"] < 5).astype(int)
    + (data["screen_time"] > 6).astype(int)
    + (data["steps"] < 5000).astype(int)
    + (data["mood"] < 3).astype(int)
)
