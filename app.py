import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px

# Load saved model
model = joblib.load("stress_model.pkl")
feature_names = joblib.load("features.pkl")
