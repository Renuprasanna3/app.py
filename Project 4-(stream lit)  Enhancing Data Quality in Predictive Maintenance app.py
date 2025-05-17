import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Streamlit page
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance Data Visualization")

# Generate synthetic dataset
def generate_dataset(seed=0):
    np.random.seed(seed)
    data = {
        "Temperature": np.random.normal(loc=70 + seed, scale=5, size=100),
        "Vibration": np.random.normal(loc=0.5 + 0.1 * seed, scale=0.05, size=100),
        "Pressure": np.random.normal(loc=30 + seed, scale=3, size=100),
        "Failure": np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    }
    return pd.DataFrame(data)

# Function to display dataset and graphs
def show_dataset_and_graphs(df, dataset_number):
    st.subheader(f"📄 Dataset {dataset_number}")
    st.dataframe(df)

    features = ["Temperature", "Vibration", "Pressure"]
    st.markdown(f"### 📊 Feature Distributions for Dataset {dataset_number}")
    for feature in features:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(df[feature], kde=True, ax=ax, color='skyblue')
        ax.set_title(f'{feature} Distribution - Dataset {dataset_number}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# Generate and show 3 datasets
for i in range(1, 4):
    df = generate_dataset(seed=i)
    show_dataset_and_graphs(df, dataset_number=i)

st.success("✅ All datasets and visualizations loaded successfully!")
