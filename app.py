import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv("motor_data.csv")

X = data[['vibration', 'temperature', 'current']]
y = data['fault']

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# UI
st.title("⚡ Smart Motor Health Monitoring System")

v = st.slider("Vibration", 0.0, 3.0, 0.5)
t = st.slider("Temperature", 0.0, 100.0, 40.0)
c = st.slider("Current", 0.0, 30.0, 10.0)

if st.button("Check Motor Health"):
    sample = pd.DataFrame([[v, t, c]],
                          columns=['vibration', 'temperature', 'current'])
    
    sample_scaled = scaler.transform(sample)
    result = model.predict(sample_scaled)

    if result[0] == 0:
        st.success("Motor is Healthy ✅")
    else:
        st.error("Fault Detected ⚠️")

        if v > 1.5:
            st.write("🔧 Bearing Fault (High Vibration)")
        if t > 65:
            st.write("🔥 Overheating / Stator Fault")
        if c > 20:
            st.write("⚡ Overload Condition")