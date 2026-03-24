import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🌍 Life Expectancy Predictor")

# Upload dataset
file = st.file_uploader("Upload Life Expectancy CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # 🔥 Fix column name issues (IMPORTANT)
    df.columns = df.columns.str.strip()

    # Show actual column names (for debugging)
    st.subheader("🧾 Column Names")
    st.write(list(df.columns))

    try:
        # Features and target
        features = ['Adult Mortality', 'Alcohol', 'GDP', 'Schooling', 'HIV/AIDS']
        target = 'Life expectancy'

        # Select required columns
        df = df[features + [target]].dropna()

        X = df[features]
        y = df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluation
        st.subheader("📊 Model Performance")
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

        # Coefficients
        st.subheader("📈 Feature Importance")
        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        })
        st.write(coef_df)

        # Plot
        st.subheader("📉 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Life Expectancy")
        ax.set_ylabel("Predicted Life Expectancy")
        st.pyplot(fig)

    except KeyError:
        st.error("❌ Column names do not match dataset. Please check above column list.")
