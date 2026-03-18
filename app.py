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

# 🔥 NEW IMPORTS
from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Feasibility Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.markdown('<h1 style="color:#4CAF50;">🤖 ML Model Feasibility Analyzer</h1>', unsafe_allow_html=True)
st.write("Upload a dataset and analyze which ML model works best.")

# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    tab1, tab2, tab3 = st.tabs(["📂 Dataset", "🤖 Model Analysis", "📊 Visualizations"])

    # ---------------- DATASET TAB ----------------
    with tab1:
        st.dataframe(df.head())

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

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            X = X.fillna(X.mean(numeric_only=True))

            try:
                y = pd.to_numeric(y)
            except:
                pass

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if y.nunique() < 10:
                problem_type = "classification"
                st.info("🔍 Detected: Classification Problem")
            else:
                problem_type = "regression"
                st.info("🔍 Detected: Regression Problem")

            if st.button("🚀 Run Model Analysis"):

                results = []

                if problem_type == "regression":

                    models = {
                        "Linear Regression": LinearRegression(),
                        "Random Forest": RandomForestRegressor(),
                        "Decision Tree": DecisionTreeRegressor()
                    }

                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)

                        results.append({
                            "Model": name,
                            "Score": r2_score(y_test, pred)
                        })

                else:

                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=2000),
                        "Random Forest": RandomForestClassifier(),
                        "Decision Tree": DecisionTreeClassifier()
                    }

                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        pred = model.predict(X_test)

                        results.append({
                            "Model": name,
                            "Score": accuracy_score(y_test, pred)
                        })

                results_df = pd.DataFrame(results)
                best_model = results_df.sort_values(by="Score", ascending=False).iloc[0]

                st.subheader("Model Comparison")
                st.dataframe(results_df)

                st.success(f"Best Model: {best_model['Model']}")

                score = best_model["Score"]

                if score > 0.85:
                    feasibility = "Highly Feasible"
                elif score > 0.65:
                    feasibility = "Moderately Feasible"
                else:
                    feasibility = "Low Feasibility"

                st.metric("Feasibility Score", f"{score*100:.2f}%")
                st.write("Conclusion:", feasibility)

                # 🎧 VOICE REPORT
                report_text = f"""
                Machine Learning Feasibility Report.
                Best model is {best_model['Model']}.
                Feasibility score is {score*100:.2f} percent.
                Conclusion: {feasibility}.
                """

                tts = gTTS(report_text, lang='en')
                tts.save("report.mp3")

                with open("report.mp3", "rb") as audio_file:
                    st.subheader("🔊 Voice Report")
                    st.audio(audio_file.read(), format="audio/mp3")

                # 📄 PDF GENERATION
                pdf_file = "report.pdf"

                doc = SimpleDocTemplate(pdf_file)
                styles = getSampleStyleSheet()

                content = []
                content.append(Paragraph("ML Feasibility Report", styles['Title']))
                content.append(Spacer(1, 10))
                content.append(Paragraph(f"Best Model: {best_model['Model']}", styles['Normal']))
                content.append(Paragraph(f"Score: {score*100:.2f}%", styles['Normal']))
                content.append(Paragraph(f"Conclusion: {feasibility}", styles['Normal']))

                doc.build(content)

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "📄 Download Report",
                        f,
                        file_name="ML_Report.pdf"
                    )

    # ---------------- VISUALIZATION TAB ----------------
    with tab3:

        if 'target' not in locals():
            st.warning("Select target column first")
        else:

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.histplot(df[target], kde=True, ax=ax2)
            st.pyplot(fig2)
