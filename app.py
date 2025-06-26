import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tle_utils import fetch_tle, compute_positions, TLE_SOURCES
from features import compute_features, compute_pairwise_features
from model import generate_synthetic_conjunction_data, train_model, predict_risk
from image_utils import create_satellite_position_image

# ----------------------------------
# Streamlit setup
# ----------------------------------
st.set_page_config(page_title="⚡ OrbitX: Real-Time Space Collision Forecaster", layout="wide")
st.sidebar.title("⚡ OrbitX: Real-Time Space Collision Forecaster")

# ----------------------------------
# Sidebar source selection
# ----------------------------------
tle_choice = st.sidebar.selectbox("🌍 Select Satellite Set", list(TLE_SOURCES.keys()))
source_url = TLE_SOURCES[tle_choice]
st.sidebar.write(f"📡 Data source: {source_url}")

# ----------------------------------
# Fetch TLE + compute positions
# ----------------------------------
tle_lines = fetch_tle(source_url)
df = compute_positions(tle_lines, max_sats=20)

# Safety check
required_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz']
if df.empty or not all(col in df.columns for col in required_columns):
    st.error("❌ No valid satellites were processed. Try another TLE source or check your connection.")
    st.stop()

# ----------------------------------
# Compute features + pairwise
# ----------------------------------
df = compute_features(df)
pair_df = compute_pairwise_features(df)
pair_df = generate_synthetic_conjunction_data(pair_df)

# ----------------------------------
# Train model + predict risk
# ----------------------------------
model, metrics = train_model(pair_df)
pair_df = predict_risk(pair_df, model)

# ----------------------------------
# Sidebar model metrics
# ----------------------------------
st.sidebar.subheader("📊 Model Performance")
st.sidebar.write(f"R² Score: {metrics['r2_score']:.3f}")
st.sidebar.write(f"RMSE: {metrics['rmse']:.3f}")
st.sidebar.write("Feature Importances:")
for feat, imp in metrics['feature_importances'].items():
    st.sidebar.write(f"- {feat}: {imp:.3f}")

# ----------------------------------
# Pillow image
# ----------------------------------
st.subheader("🛰 Satellite Positions (x-y Projection Image)")
img = create_satellite_position_image(df)
st.image(img, caption="Satellite positions (x-y view, Pillow)", use_container_width=True)

# ----------------------------------
# 3D plot
# ----------------------------------
fig = go.Figure(go.Scatter3d(
    x=df['x'], y=df['y'], z=df['z'],
    mode='markers',
    marker=dict(size=5, color='cyan', opacity=0.8),
    text=df['name']
))
fig.update_layout(
    title="Satellite Positions in 3D",
    scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# Risky pairs table
# ----------------------------------
st.subheader("🚨 At-Risk Satellite Pairs (Top 10)")
st.dataframe(pair_df.sort_values(by="risk_score", ascending=False)[
    ['name1', 'name2', 'distance', 'altitude_diff', 'speed_diff', 'risk_score']
].head(10))
