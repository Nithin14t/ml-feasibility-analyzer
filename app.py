import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

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

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        target = st.selectbox("Select Target Column", df.columns)

    # ---------------- MODEL ANALYSIS TAB ----------------
    with tab2:

        if 'target' not in locals():
            st.warning("Please select target column in Dataset tab")
        else:

            X = df.drop(columns=[target])
            y = df[target]

            # Handle categorical
            X = pd.get_dummies(X)

            # Handle missing values
            X = X.fillna(X.mean(numeric_only=True))

            # Try numeric conversion
            try:
                y = pd.to_numeric(y)
            except:
                pass

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 🔥 AUTO DETECT
            if y.nunique() < 10:
                problem_type = "classification"
                st.info("🔍 Detected: Classification Problem")
            else:
                problem_type = "regression"
                st.info("🔍 Detected: Regression Problem")

            if st.button("🚀 Run Model Analysis"):

                results = []

                with st.spinner("Training models..."):

                    if problem_type == "regression":

                        models = {
                            "Linear Regression": LinearRegression(),
                            "Random Forest Regressor": RandomForestRegressor(),
                            "Decision Tree Regressor": DecisionTreeRegressor()
                        }

                        for name, model in models.items():
                            try:
                                model.fit(X_train, y_train)
                                pred = model.predict(X_test)

                                results.append({
                                    "Model": name,
                                    "R2 Score": round(r2_score(y_test, pred), 3),
                                    "MAE": round(mean_absolute_error(y_test, pred), 3),
                                    "CV Score": round(cross_val_score(model, X, y, cv=5).mean(), 3)
                                })
                            except Exception as e:
                                st.warning(f"{name} failed: {e}")

                        results_df = pd.DataFrame(results)
                        best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]
                        score = best_model["R2 Score"]

                    else:  # CLASSIFICATION

                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=2000),
                            "Random Forest Classifier": RandomForestClassifier(),
                            "Decision Tree Classifier": DecisionTreeClassifier()
                        }

                        for name, model in models.items():
                            try:
                                model.fit(X_train, y_train)
                                pred = model.predict(X_test)

                                results.append({
                                    "Model": name,
                                    "Accuracy": round(accuracy_score(y_test, pred), 3),
                                    "CV Score": round(cross_val_score(model, X, y, cv=5).mean(), 3)
                                })
                            except Exception as e:
                                st.warning(f"{name} failed: {e}")

                        results_df = pd.DataFrame(results)
                        best_model = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
                        score = best_model["Accuracy"]

                if len(results) == 0:
                    st.error("All models failed.")
                    st.stop()

                st.subheader("Model Comparison")
                st.dataframe(results_df)

                st.success(f"Best Model: {best_model['Model']}")

                # 🔥 Feasibility Logic
                if score > 0.85:
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
            sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, ax=ax)
            st.pyplot(fig)

            st.subheader("Target Distribution")

            fig2, ax2 = plt.subplots()
            sns.histplot(df[target], kde=True, ax=ax2)
            st.pyplot(fig2)
