# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

# Load sample CSV from local file
@st.cache_data
def load_local_data():
    return pd.read_csv("sample_data.csv")  # Place the file in the same folder as app.py

# Clean the data
def clean_sensor_data(df):
    df_clean = df.copy()
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    for col in df_clean.select_dtypes(include=np.number).columns:
        z_scores = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        df_clean = df_clean[(np.abs(z_scores) < 3)]
    return df_clean

# Train model
def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    return model, pd.DataFrame(report).transpose()

# UI setup
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")
st.title("🔧 AI-based Predictive Maintenance System with Data Quality Enhancement")

# Auto-load data
df = load_local_data()
st.subheader("📄 Raw Sensor Data")
st.write(df.head())
st.write(f"🔢 Dataset shape: {df.shape}")

df_clean = clean_sensor_data(df)
st.subheader("🧼 Cleaned Sensor Data")
st.write(df_clean.head())
st.write(f"✅ Cleaned shape: {df_clean.shape}")

# Feature distribution
st.subheader("📊 Feature Distributions")
num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
if num_cols:
    selected_feature = st.selectbox("Select feature to visualize", num_cols)
    fig = px.histogram(df_clean, x=selected_feature, nbins=30, title=f"Distribution of {selected_feature}")
    st.plotly_chart(fig)

# Correlation heatmap
st.subheader("📌 Correlation Heatmap")
corr = df_clean.select_dtypes(include=np.number).corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Train model
if 'failure' in df_clean.columns:
    st.subheader("🤖 Model Training and Evaluation")
    model, report_df = train_model(df_clean, target_column='failure')
    st.dataframe(report_df)

    st.subheader("📈 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=df_clean.drop(columns=['failure']).columns)
    fig_imp = px.bar(importances.sort_values(ascending=False), title="Feature Importance", labels={'value': 'Importance', 'index': 'Feature'})
    st.plotly_chart(fig_imp)
else:
    st.warning("⚠️ The dataset must contain a 'failure' column as the target.")
