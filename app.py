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

# User Inputs
st.subheader("📋 Enter Your Daily Lifestyle Metrics:")
sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
water = st.slider("Water Intake (glasses)", 1, 12, 6)
screen = st.slider("Screen Time (hours)", 1.0, 12.0, 6.0)
steps = st.slider("Steps Walked", 1000, 20000, 7000, step=500)
mood = st.select_slider("Mood", options=[1,2,3,4,5], value=3)