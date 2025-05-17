import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load sample CSV from local file
@st.cache_data
def load_data():
    return pd.read_csv("sample_data.csv")

# Clean data (fill NAs and remove outliers)
def clean_sensor_data(df):
    df_clean = df.copy()
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

    # Remove outliers
    for col in df_clean.select_dtypes(include=np.number).columns:
        z = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        df_clean = df_clean[np.abs(z) < 3]
    
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

df = load_data()
st.subheader("Raw Sensor Data")
st.write(df.head())

df_clean = clean_sensor_data(df)
st.subheader("Cleaned Sensor Data")
st.write(df_clean.head())

if 'failure' in df_clean.columns:
    st.subheader("Model Training and Evaluation")
    model, report_df = train_model(df_clean, target_column='failure')
    st.dataframe(report_df)

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": df_clean.drop(columns=["failure"]).columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    feature_options = [col for col in df_clean.columns if df_clean[col].dtype != 'object' and col != 'failure']
    selected_feature = st.selectbox("Select feature to visualize", feature_options)

    if selected_feature:
        fig2, ax2 = plt.subplots()
        sns.histplot(df_clean[selected_feature], kde=True, ax=ax2)
        st.pyplot(fig2)
else:
    st.warning("⚠️ 'failure' column not found in dataset.")

  
