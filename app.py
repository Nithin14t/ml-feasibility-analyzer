import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- UI ----------------
st.set_page_config(page_title="ML AI Advisor", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#4CAF50;'>🚀 ML AI Advisor</h1>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Controls")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------- MAIN ----------------
if file:

    df = pd.read_csv(file)

    tab1, tab2, tab3 = st.tabs(["📂 Dataset", "🤖 Analysis", "📊 Visuals"])

    # -------- TAB 1 --------
    with tab1:
        st.dataframe(df.head())
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])
        target = st.selectbox("Target Column", df.columns)

    # -------- TAB 2 --------
    with tab2:

        if 'target' not in locals():
            st.warning("Select target")
        else:

            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]

            X = X.fillna(X.mean(numeric_only=True))

            try:
                y = pd.to_numeric(y)
            except:
                pass

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            problem = "classification" if y.nunique() < 10 else "regression"
            st.info(f"Detected: {problem}")

            if st.button("Run Analysis"):

                results = []

                if problem == "regression":
                    models = {
                        "Linear": LinearRegression(),
                        "Forest": RandomForestRegressor(),
                        "Tree": DecisionTreeRegressor()
                    }

                    for n, m in models.items():
                        m.fit(X_train, y_train)
                        pred = m.predict(X_test)
                        results.append({"Model": n, "Score": r2_score(y_test, pred)})

                else:
                    models = {
                        "Logistic": LogisticRegression(max_iter=2000),
                        "Forest": RandomForestClassifier(),
                        "Tree": DecisionTreeClassifier()
                    }

                    for n, m in models.items():
                        m.fit(X_train, y_train)
                        pred = m.predict(X_test)
                        results.append({"Model": n, "Score": accuracy_score(y_test, pred)})

                results_df = pd.DataFrame(results)
                best = results_df.sort_values(by="Score", ascending=False).iloc[0]

                st.dataframe(results_df)
                st.success(f"Best Model: {best['Model']}")

                score = best["Score"]

                if score > 0.85:
                    feasibility = "Highly Feasible"
                elif score > 0.65:
                    feasibility = "Moderate"
                else:
                    feasibility = "Low"

                st.metric("Score", f"{score*100:.2f}%")

                # 🤖 AI ADVISOR
                st.subheader("🤖 AI Advisor")
                if "Forest" in best["Model"]:
                    st.write("Model captured complex patterns. Data likely non-linear.")
                elif "Linear" in best["Model"]:
                    st.write("Data shows linear relationships.")
                else:
                    st.write("Model fits simple structure. Consider feature engineering.")

                # 🎧 VOICE
                text = f"Best model is {best['Model']}. Score is {score*100:.1f} percent. Feasibility is {feasibility}."
                tts = gTTS(text)
                tts.save("voice.mp3")
                st.audio("voice.mp3")

                # -------- CHARTS FOR PDF --------
                heatmap_path = "heatmap.png"
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(numeric_only=True), ax=ax)
                plt.savefig(heatmap_path)
                plt.close()

                dist_path = "dist.png"
                fig2, ax2 = plt.subplots()
                sns.histplot(df[target], ax=ax2)
                plt.savefig(dist_path)
                plt.close()

                # -------- PDF --------
                pdf = "report.pdf"
                doc = SimpleDocTemplate(pdf)
                styles = getSampleStyleSheet()

                content = []
                content.append(Paragraph("ML Report", styles['Title']))
                content.append(Spacer(1, 10))
                content.append(Paragraph(f"Best Model: {best['Model']}", styles['Normal']))
                content.append(Paragraph(f"Score: {score*100:.2f}%", styles['Normal']))
                content.append(Paragraph(f"Feasibility: {feasibility}", styles['Normal']))

                content.append(Spacer(1, 10))
                content.append(Image(heatmap_path, width=400, height=300))
                content.append(Image(dist_path, width=400, height=300))

                doc.build(content)
                # 📄 PDF DOWNLOAD (FIXED)
                with open(pdf_file, "rb") as f:
                      st.download_button(
                      label="📄 Download Report",
                      data=f.read(),
                      file_name="ML_Report.pdf",
                       mime="application/pdf"
                                              )



    # -------- TAB 3 --------
    with tab3:
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), ax=ax)
        st.pyplot(fig)
