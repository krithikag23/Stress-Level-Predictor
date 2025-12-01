import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px

# Load saved model
model = joblib.load("stress_model.pkl")
feature_names = joblib.load("features.pkl")

st.set_page_config(page_title="Stress Predictor AI", page_icon="🧠", layout="centered")
st.title("🧠 Stress Level Predictor")
st.caption("Analyze lifestyle factors to estimate stress level and get improvement tips!")