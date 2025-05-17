import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set Streamlit page config
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("sample_data.csv")
    return df

# Clean data
def clean_sensor_data(df):
    df_clean = df.copy()
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    for col in df_clean.select_dtypes(include=np.number).columns:
        z = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        df_clean = df_clean[np.abs(z) < 3]
    return df_clean

# Train model and return results
def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return model, pd.DataFrame(report).transpose()

# App start
st.title("🔧 Predictive Maintenance with Enhanced Data Quality")

df = load_data()
st.subheader("📊 Raw Data")
st.dataframe(df.head())

# Clean the data
df_clean = clean_sensor_data(df)
st.subheader("🧼 Cleaned Data")
st.dataframe(df_clean.head())

# Model Training
if 'failure' in df_clean.columns:
    st.subheader("🤖 Model Evaluation")
    model, report_df = train_model(df_clean, 'failure')
    st.dataframe(report_df)

    # Feature importance
    st.subheader("📈 Feature Importance")
    feature_importances = model.feature_importances_
    features = df_clean.drop(columns=['failure']).columns
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by="Importance", ascending=False)

    fig1, ax1 = plt.subplots()
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax1)
    st.pyplot(fig1)

    # Feature Distributions
    st.subheader("🔍 Feature Distribution Plot")
    numerical_cols = [col for col in df_clean.columns if df_clean[col].dtype != 'object' and col != 'failure']
    selected_feature = st.selectbox("Select a numerical feature", numerical_cols)

    if selected_feature:
        fig2, ax2 = plt.subplots()
        sns.histplot(df_clean[selected_feature], kde=True, ax=ax2)
        st.pyplot(fig2)
else:
    st.error("The 'failure' column is missing from the dataset.")


  
