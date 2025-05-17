# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load local data
@st.cache_data
def load_local_data():
    return pd.read_csv("sample_data.csv")  # Ensure this file exists in the same directory

# Clean data
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

# Streamlit UI
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")
st.title("🔧 AI-based Predictive Maintenance System with Data Quality Enhancement")

df = load_local_data()
st.subheader("📄 Raw Sensor Data")
st.write(df.head())
st.write(f"🔢 Dataset shape: {df.shape}")

df_clean = clean_sensor_data(df)
st.subheader("🧼 Cleaned Sensor Data")
st.write(df_clean.head())
st.write(f"✅ Cleaned shape: {df_clean.shape}")

# Distribution plot
st.subheader("📊 Feature Distributions")
num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
selected_feature = st.selectbox("Select feature to visualize", num_cols)
fig, ax = plt.subplots()
sns.histplot(df_clean[selected_feature], kde=True, ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.subheader("📌 Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
st.pyplot(fig2)

# Model Training
if 'failure' in df_clean.columns:
    st.subheader("🤖 Model Training and Evaluation")
    model, report_df = train_model(df_clean, target_column='failure')
    st.dataframe(report_df)

    # Feature importance
    st.subheader("📈 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=df_clean.drop(columns=['failure']).columns)
    fig3, ax3 = plt.subplots()
    importances.sort_values().plot(kind='barh', ax=ax3)
    st.pyplot(fig3)
else:
    st.warning("⚠️ The dataset must contain a 'failure' column.")
