# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("🔧 Predictive Maintenance: Data Quality & Failure Prediction")

# Sidebar seed input
seed = st.sidebar.number_input("Random Seed", value=42)

# Generate synthetic data
np.random.seed(seed)
n_samples = 1000
df = pd.DataFrame({
    'temperature': np.random.normal(75, 10, n_samples),
    'pressure': np.random.normal(30, 5, n_samples),
    'vibration': np.random.normal(0.5, 0.1, n_samples),
    'failure': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
})

# Inject missing values and outliers
for col in ['temperature', 'pressure']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan
df.loc[df.sample(frac=0.02).index, 'vibration'] = 3.0

st.subheader("📊 Raw Data (with Missing and Outliers)")
st.dataframe(df.head())

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[['temperature', 'pressure']] = imputer.fit_transform(df[['temperature', 'pressure']])

# Outlier detection
features = ['temperature', 'pressure', 'vibration']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
iso = IsolationForest(contamination=0.02, random_state=seed)
df['outlier'] = iso.fit_predict(X_scaled)
df_clean = df[df['outlier'] == 1].drop(columns='outlier')

# Train-test split and SMOTE
X = df_clean[features]
y = df_clean['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed)
smote = SMOTE(random_state=seed)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=seed)
model.fit(X_train_bal, y_train_bal)
y_pred = model.predict(X_test)

# Display metrics
st.subheader("📈 Classification Report")
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Feature importance
importances = pd.Series(model.feature_importances_, index=features)
st.subheader("📌 Feature Importance")
st.bar_chart(importances)




