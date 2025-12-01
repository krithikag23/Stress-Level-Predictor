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