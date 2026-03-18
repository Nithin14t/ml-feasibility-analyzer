import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Feasibility Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main-title{
font-size:40px;
font-weight:bold;
color:#4CAF50;
}

.metric-box{
background-color:#111827;
padding:20px;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤖 ML Model Feasibility Analyzer</p>', unsafe_allow_html=True)
st.write("Upload a dataset and analyze which ML model works best.")

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    tab1, tab2, tab3 = st.tabs(["📂 Dataset", "🤖 Model Analysis", "📊 Visualizations"])

    # ---------------- DATASET TAB ----------------
    with tab1:

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Health")

        col1, col2, col3 = st.columns(3)

        rows = df.shape[0]
        cols = df.shape[1]
        missing = df.isnull().sum().sum()

        col1.metric("Rows", rows)
        col2.metric("Columns", cols)
        col3.metric("Missing Values", missing)

        target = st.selectbox("Select Target Column", df.columns)

    # ---------------- MODEL ANALYSIS TAB ----------------
    with tab2:

        if 'target' not in locals():
            st.warning("Please select target column in Dataset tab")
        else:

            X = df.drop(columns=[target])
            y = df[target]

            # Handle categorical features
            X = pd.get_dummies(X)

            # 🔥 FIX 1: Handle missing values
            X = X.fillna(X.mean(numeric_only=True))
            if y.dtype != 'object':
                y = y.fillna(y.mean())

            # 🔥 FIX 2: Ensure target is numeric (for regression)
            try:
                y = pd.to_numeric(y)
            except:
                st.error("Target column must be numeric for regression.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if st.button("🚀 Run Model Analysis"):

                with st.spinner("Training models..."):

                    models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)
                    }

                    results = []

                    for name, model in models.items():
                        try:
                            model.fit(X_train, y_train)

                            train_pred = model.predict(X_train)
                            test_pred = model.predict(X_test)

                            train_r2 = r2_score(y_train, train_pred)
                            test_r2 = r2_score(y_test, test_pred)

                            mae = mean_absolute_error(y_test, test_pred)

                            cv = cross_val_score(model, X, y, cv=5).mean()

                            results.append({
                                "Model": name,
                                "Train R2": round(train_r2, 3),
                                "Test R2": round(test_r2, 3),
                                "MAE": round(mae, 3),
                                "CV Score": round(cv, 3)
                            })

                        except Exception as e:
                            st.warning(f"{name} failed: {e}")

                    if len(results) == 0:
                        st.error("All models failed. Check dataset.")
                        st.stop()

                    results_df = pd.DataFrame(results)

                    st.subheader("Model Comparison")
                    st.dataframe(results_df)

                    best_model = results_df.sort_values(by="Test R2", ascending=False).iloc[0]

                    st.success(f"Best Model: {best_model['Model']}")

                    score = best_model["Test R2"]
                    gap = abs(best_model["Train R2"] - best_model["Test R2"])

                    if score > 0.85 and gap < 0.1:
                        feasibility = "Highly Feasible"
                    elif score > 0.65:
                        feasibility = "Moderately Feasible"
                    else:
                        feasibility = "Low Feasibility"

                    st.metric("Feasibility Score", f"{score*100:.2f}%")
                    st.progress(max(0, min(1, score)))
                    st.write("Conclusion:", feasibility)

    # ---------------- VISUALIZATION TAB ----------------
    with tab3:

        if 'target' not in locals():
            st.warning("Select target column first")
        else:

            st.subheader("Feature Correlation")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                df.corr(numeric_only=True),
                cmap="coolwarm",
                annot=True,
                ax=ax
            )
            st.pyplot(fig)

            st.subheader("Target Distribution")

            fig2, ax2 = plt.subplots()
            sns.histplot(df[target], kde=True, ax=ax2)
            st.pyplot(fig2)
