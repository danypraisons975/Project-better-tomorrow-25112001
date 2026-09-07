import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import altair as alt
import os

st.set_page_config(page_title="Smart Waste Collection Management", layout="wide")

# --- Helpers to convert time strings <-> minutes since midnight ---
def time_to_minutes(t):
    if pd.isna(t):
        return None
    t = str(t).strip()
    if t.lower() == "unknown" or t == "":
        return None
    try:
        parts = t.split(":")
        h = int(parts[0])
        m = int(parts[1])
        return h * 60 + m
    except Exception:
        return None

def minutes_to_time(m):
    if m is None or np.isnan(m):
        return "Unknown"
    m = int(round(m))
    h = m // 60
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

# --- Load data ---
DATA_PATH = os.path.join("data", "sample_data.csv")
df = pd.read_csv(DATA_PATH)
# ensure columns trimmed
df.columns = [c.strip() for c in df.columns]

df["ArrivalMinutes"] = df["Arrival Time"].apply(time_to_minutes)

# --- Prepare training data (drop unknown arrival rows) ---
train_df = df[df["ArrivalMinutes"].notna()].copy()
X = train_df[["Day", "Houses Waiting", "Waste Ready"]]
y = train_df["ArrivalMinutes"]

# Train a Random Forest Regressor
if len(train_df) >= 3:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
else:
    model = None
    mae = None

# --- Prediction target: if there's a row with 'Unknown', predict for it; else let user pick a day to predict ---
unknown_rows = df[df["ArrivalMinutes"].isna()]
if not unknown_rows.empty:
    predict_row = unknown_rows.iloc[0]
    pred_day = int(predict_row["Day"])
    pred_houses_waiting = int(predict_row["Houses Waiting"])
    pred_waste_ready = int(predict_row["Waste Ready"])
    sidebar_msg = f"Predicting for Day {pred_day} (from data row with 'Unknown' arrival)."
else:
    # default: next day
    pred_day = int(df["Day"].max() + 1)
    pred_houses_waiting = 12
    pred_waste_ready = int(df["Waste Ready"].median())
    sidebar_msg = f"No 'Unknown' arrival in data — default predict for next day ({pred_day})."

# Sidebar controls
st.sidebar.header("Prediction Input")
st.sidebar.write(sidebar_msg)
day_input = st.sidebar.number_input("Day", min_value=1, value=pred_day, step=1)
houses_input = st.sidebar.number_input("Houses Waiting", min_value=1, value=pred_houses_waiting, step=1)
waste_ready_input = st.sidebar.number_input("Waste Ready", min_value=0, value=pred_waste_ready, step=1)
st.sidebar.markdown("---")
st.sidebar.write("Model info")
if model is not None:
    st.sidebar.write("RandomForestRegressor (trained on available historical rows)")
    st.sidebar.write(f"Validation MAE: {mae:.1f} minutes")
else:
    st.sidebar.write("Not enough data to train model.")

# --- Do prediction ---
pred_minutes = None
pred_time_str = "Unknown"
status = "Unknown"

if model is not None:
    X_pred = np.array([[int(day_input), int(houses_input), int(waste_ready_input)]])
    pred_minutes = model.predict(X_pred)[0]
    pred_time_str = minutes_to_time(pred_minutes)
    status = "On-time (before 09:00 AM)" if pred_minutes <= 9 * 60 else "Late (after 09:00 AM)"

# --- Layout: Dashboard metrics ---
st.title("Smart Waste Collection Management System")
st.markdown*
