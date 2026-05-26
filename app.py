# =========================================================
# KNN REGRESSOR - DIABETES DATASET
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="KNN Regressor Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# LOAD CUSTOM CSS
# =========================================================

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# =========================================================
# TITLE
# =========================================================

st.title("📈 KNN Regressor - Diabetes Dataset")

st.markdown("""
This application demonstrates regression using K-Nearest Neighbors Regressor.

### Workflow Included
- Dataset Loading
- Data Cleaning
- Exploratory Data Analysis
- Feature Scaling
- Save Preprocessed Dataset
- Model Training
- Prediction
- Model Evaluation
""")

# =========================================================
# LOAD DATASET
# =========================================================

st.header("📂 Step 1 : Load Dataset")

data = load_diabetes()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# =========================================================
# DATASET METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Features", len(data.feature_names))

# =========================================================
# DATA CLEANING
# =========================================================

st.header("🧹 Step 2 : Data Cleaning")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

with col2:

    st.subheader("Duplicate Rows")

    duplicates = df.duplicated().sum()

    st.write(f"Duplicate Rows : {duplicates}")

st.success("✅ Data Cleaning Completed Successfully")

# =========================================================
# FEATURES & TARGET
# =========================================================

st.header("⚙️ Step 3 : Feature Selection")

X = df.drop("target", axis=1)
y = df["target"]

st.write("### Independent Features")
st.write(list(X.columns))

st.write("### Target Variable")
st.write("Disease Progression")

# =========================================================
# FEATURE SCALING
# =========================================================

st.header("📏 Step 4 : Feature Scaling")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Convert scaled data into dataframe

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

st.success("✅ Feature Scaling Applied Successfully")

# =========================================================
# SAVE PREPROCESSED DATA
# =========================================================

st.header("💾 Step 5 : Save Preprocessed Dataset")

# Create data folder if not exists

os.makedirs("data", exist_ok=True)

# Combine scaled features and target

processed_df = X_scaled_df.copy()

processed_df["target"] = y.values

if st.button("Save Preprocessed Data"):

    processed_df.to_csv(
        "data/preprocessed_diabetes.csv",
        index=False
    )

    st.success(
        "✅ Preprocessed dataset saved successfully in data/ folder"
    )

# =========================================================
# EXPLORATORY DATA ANALYSIS
# =========================================================

st.header("📊 Step 6 : Exploratory Data Analysis")

# =========================================================
# STATISTICAL SUMMARY
# =========================================================

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

st.subheader("Correlation Heatmap")

fig1, ax1 = plt.subplots(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax1
)

st.pyplot(fig1)

# =========================================================
# HISTOGRAM
# =========================================================

st.subheader("Target Variable Distribution")

fig2, ax2 = plt.subplots(figsize=(8, 5))

sns.histplot(
    df["target"],
    bins=30,
    kde=True,
    color="blue",
    ax=ax2
)

ax2.set_xlabel("Target Value")
ax2.set_ylabel("Frequency")

st.pyplot(fig2)

# =========================================================
# FEATURE VS TARGET SCATTER PLOT
# =========================================================

st.subheader("Feature vs Target")

selected_feature = st.selectbox(
    "Select Feature",
    X.columns
)

fig3, ax3 = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    x=df[selected_feature],
    y=df["target"],
    color="green",
    ax=ax3
)

ax3.set_xlabel(selected_feature)
ax3.set_ylabel("Target")

st.pyplot(fig3)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

st.header("✂️ Step 7 : Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Samples",
        X_train.shape[0]
    )

with col2:
    st.metric(
        "Testing Samples",
        X_test.shape[0]
    )

# =========================================================
# MODEL TRAINING
# =========================================================

st.header("🤖 Step 8 : Model Training")

k = st.slider(
    "Select K Value",
    1,
    20,
    5
)

model = KNeighborsRegressor(
    n_neighbors=k
)

model.fit(X_train, y_train)

st.success("✅ KNN Regressor Model Trained Successfully")

# =========================================================
# PREDICTIONS
# =========================================================

st.header("📌 Step 9 : Predictions")

y_pred = model.predict(X_test)

prediction_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.dataframe(
    prediction_df.head(10),
    use_container_width=True
)

# =========================================================
# MODEL EVALUATION
# =========================================================

st.header("📉 Step 10 : Model Evaluation")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("MSE", f"{mse:.2f}")

with col3:
    st.metric("R2 Score", f"{r2:.2f}")

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader("Actual vs Predicted Values")

fig4, ax4 = plt.subplots(figsize=(8, 5))

ax4.scatter(
    y_test,
    y_pred
)

ax4.set_xlabel("Actual Values")
ax4.set_ylabel("Predicted Values")

st.pyplot(fig4)

# =========================================================
# RESIDUAL PLOT
# =========================================================

st.subheader("Residual Plot")

residuals = y_test - y_pred

fig5, ax5 = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals,
    color="red",
    ax=ax5
)

ax5.axhline(
    y=0,
    linestyle="--"
)

ax5.set_xlabel("Predicted Values")
ax5.set_ylabel("Residuals")

st.pyplot(fig5)

# =========================================================
# USER INPUT
# =========================================================

st.header("🎯 Step 11 : Predict Diabetes Progression")

user_input = []

col1, col2 = st.columns(2)

columns = list(X.columns)

for i in range(len(columns)):

    with col1 if i % 2 == 0 else col2:

        val = st.number_input(
            f"Enter {columns[i]}",
            value=float(df[columns[i]].mean())
        )

        user_input.append(val)

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict"):

    scaled_data = scaler.transform([user_input])

    prediction = model.predict(scaled_data)[0]

    st.markdown(
        f"""
        <div style="
            background-color:#1e3a8a;
            padding:20px;
            border-radius:10px;
            text-align:center;
            color:white;
            font-size:24px;
        ">
            Predicted Diabetes Progression : {prediction:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )