import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data from GitHub
def load_data_from_github(github_raw_url):
    try:
        response = requests.get(github_raw_url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            st.error("Error loading data from GitHub.")
            return None
    except Exception as e:
        st.error(f"Exception occurred: {e}")
        return None

# Clean and validate data
def clean_sensor_data(df):
    df_clean = df.copy()
    # Example: Fill missing values with median
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    # Remove outliers using Z-score method
    for col in df_clean.select_dtypes(include=np.number).columns:
        z_scores = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        df_clean = df_clean[(np.abs(z_scores) < 3)]
    return df_clean

# Train and predict model
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

github_url = st.text_input("Enter GitHub Raw CSV URL", 
                           "https://raw.githubusercontent.com/your-username/your-repo/main/sensor_data.csv")

if st.button("Load and Clean Data"):
    df = load_data_from_github(github_url)
    if df is not None:
        st.subheader("Raw Sensor Data")
        st.write(df.head())

        st.subheader("Cleaned Sensor Data")
        df_clean = clean_sensor_data(df)
        st.write(df_clean.head())

        if 'failure' in df_clean.columns:
            st.subheader("Model Training and Evaluation")
            model, report_df = train_model(df_clean, target_column='failure')
            st.dataframe(report_df)
        else:
            st.warning("The dataset must contain a 'failure' column as the target.")
