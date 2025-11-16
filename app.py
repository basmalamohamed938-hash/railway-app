import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from multi_model import MultiTaskRailwayModel  # your previously created class

# -----------------------------
# Load trained multi-task model
model = MultiTaskRailwayModel()
model.load(path='models/')

st.set_page_config(page_title="🚆 Railway Prediction App", layout="wide")
st.title("🚆 Railway Prediction App")
st.write("Predict **Refund Request**, **Price**, and **Journey Status** for railway tickets.")

# -----------------------------
# Sidebar: User inputs
st.sidebar.header("Passenger / Ticket Info")

input_data = {}

# Generate inputs for categorical columns dynamically
for col, le in model.label_encoders.items():
    input_data[col] = st.sidebar.selectbox(col, le.classes_)

# Numeric columns (adjust to your dataset)
numeric_cols = ['Distance', 'Ticket Price']  # replace/add actual numeric columns
for col in numeric_cols:
    input_data[col] = st.sidebar.number_input(col, min_value=0, step=1)

# Convert to DataFrame for prediction
input_df = pd.DataFrame([input_data])

# -----------------------------
# Batch prediction via CSV
st.subheader("Or upload CSV for batch predictions")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Batch predictions will be based on uploaded CSV.")
else:
    batch_df = None

# -----------------------------
# Prediction button
if st.button("Predict"):
    # Choose input data: single row or batch
    data_to_predict = batch_df if batch_df is not None else input_df
    preds = model.predict(data_to_predict)

    st.subheader("Predictions")
    if batch_df is not None:
        st.write(preds)  # Show batch predictions
    else:
        # Color-coded display for single row
        st.markdown(f"**Refund Request:** :green[{preds['Refund Request'][0]}]")
        st.markdown(f"**Price:** :blue[{preds['Price'][0]:.2f}]")
        st.markdown(f"**Journey Status:** :orange[{preds['Journey Status'][0]}]")

# -----------------------------
# Optional: Feature importance visualization for Journey Status
st.subheader("Feature Importance (Journey Status)")
feat_importance = model.models['Journey Status'].feature_importances_
feature_names = input_df.columns.tolist()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feature_names, feat_importance, color='skyblue')
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
st.pyplot(fig)

# -----------------------------
# Footer / Info
st.info("This app predicts Refund Requests, Ticket Price, and Journey Status using a multi-task Random Forest model.")
