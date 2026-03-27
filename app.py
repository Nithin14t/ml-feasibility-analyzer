import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import json
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ML AI Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

USERS_FILE = "users.json"

# ─────────────────────────────────────────
#  USER STORE  (JSON-based, file-persisted)
# ─────────────────────────────────────────
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password, email, full_name):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "full_name": full_name
    }
    save_users(users)
    return True, "Account created successfully!"

def authenticate(username, password):
    users = load_users()
    if username not in users:
        return False, "User not found. Please sign up."
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, users[username]["full_name"]

# ─────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "username": "",
    "full_name": "",
    "auth_page": "login",       # "login" | "signup"
    "chat_history": [],
    "analysis_done": False,
    "analysis_summary": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] *        { color: #e6edf3 !important; }

/* ── Cards ── */
.auth-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 36px 40px;
    max-width: 440px;
    margin: 40px auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.auth-title {
    text-align: center;
    color: #58a6ff;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.auth-subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 0.88rem;
    margin-bottom: 28px;
}

/* ── Chat bubble ── */
.bubble-user {
    background: #1f6feb;
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 6px 0 6px 60px;
    font-size: 0.9rem;
    line-height: 1.5;
}
.bubble-bot {
    background: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    margin: 6px 60px 6px 0;
    font-size: 0.9rem;
    line-height: 1.5;
}
.chat-name-user { text-align:right; font-size:0.72rem; color:#8b949e; margin-right:4px; margin-bottom:2px; }
.chat-name-bot  { font-size:0.72rem; color:#58a6ff; margin-left:4px; margin-bottom:2px; }
.chat-wrap { max-height: 420px; overflow-y: auto; padding: 8px 4px; }

/* ── Tabs ── */
[data-baseweb="tab"] { color: #8b949e !important; }
[aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* ── Metrics ── */
[data-testid="stMetric"] label { color: #8b949e !important; }
[data-testid="stMetricValue"]  { color: #3fb950 !important; font-weight: 700; }

/* ── Inputs ── */
input, textarea, [data-baseweb="input"] input {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header h1 { color: #58a6ff; margin: 0; font-size: 1.6rem; }
.app-header p  { color: #8b949e; margin: 0; font-size: 0.85rem; }
.user-chip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 100px;
    padding: 6px 16px;
    color: #3fb950;
    font-size: 0.82rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  AUTH PAGES
# ═══════════════════════════════════════════
def show_login():
    st.markdown("""
    <div class="auth-card">
      <div class="auth-title">🔐 Welcome Back</div>
      <div class="auth-subtitle">Sign in to ML AI Advisor</div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        with st.container():
            st.markdown("### 🔐 Sign In")
            username = st.text_input("👤 Username", key="login_user", placeholder="Enter username")
            password = st.text_input("🔒 Password", type="password", key="login_pass", placeholder="Enter password")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 Login", use_container_width=True):
                    if not username or not password:
                        st.error("Please fill all fields.")
                    else:
                        ok, result = authenticate(username, password)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.full_name = result
                            st.success(f"Welcome, {result}! 🎉")
                            time.sleep(0.8)
                            st.rerun()
                        else:
                            st.error(result)
            with c2:
                if st.button("📝 Sign Up", use_container_width=True):
                    st.session_state.auth_page = "signup"
                    st.rerun()

            st.markdown("---")
            st.caption("New here? Click **Sign Up** to create a free account.")


def show_signup():
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("### 📝 Create Account")
        full_name = st.text_input("👤 Full Name",     key="reg_name",  placeholder="Your full name")
        email     = st.text_input("📧 Email",         key="reg_email", placeholder="your@email.com")
        username  = st.text_input("🆔 Username",      key="reg_user",  placeholder="Choose a username")
        password  = st.text_input("🔒 Password",      type="password", key="reg_pass",  placeholder="Min 6 characters")
        confirm   = st.text_input("🔒 Confirm Password", type="password", key="reg_conf", placeholder="Repeat password")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Create Account", use_container_width=True):
                if not all([full_name, email, username, password, confirm]):
                    st.error("Please fill all fields.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(username, password, email, full_name)
                    if ok:
                        st.success(msg + " Redirecting to login…")
                        time.sleep(1.2)
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(msg)
        with c2:
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()

        st.caption("Already have an account? Click **Back to Login**.")


# ═══════════════════════════════════════════
#  CHATBOT LOGIC  (rule-based + Claude API)
# ═══════════════════════════════════════════
ML_KB = {
    "linear regression": "**Linear Regression** models the relationship between features and a continuous target using a straight line (y = mx + c). Best when data has a linear pattern. Metric: R² score.",
    "logistic regression": "**Logistic Regression** is used for classification tasks. Despite its name, it predicts probabilities using the sigmoid function and outputs classes. Works well for linearly separable data.",
    "random forest": "**Random Forest** builds many decision trees and averages their output. It handles non-linear data, is robust to overfitting, and works well even without much tuning.",
    "decision tree": "**Decision Tree** splits data using feature thresholds to form a tree. Easy to interpret but can overfit — use `max_depth` to regularise.",
    "r2 score": "**R² (R-squared)** measures how much variance in the target is explained by the model. R²=1.0 is perfect; R²=0 means the model is no better than predicting the mean.",
    "accuracy": "**Accuracy** = correct predictions / total predictions. Useful for balanced datasets, but misleading when classes are imbalanced — consider F1 score or AUC in that case.",
    "overfitting": "**Overfitting** happens when a model learns the training data too well (including noise) and performs poorly on new data. Fix: regularisation, more data, pruning, or cross-validation.",
    "underfitting": "**Underfitting** means the model is too simple to capture the underlying patterns. Fix: use a more complex model, add features, or reduce regularisation.",
    "train test split": "**Train-Test Split** divides data into training (80%) and testing (20%) sets. The model learns from training data and is evaluated on unseen test data.",
    "feature engineering": "**Feature Engineering** is the process of creating new features from raw data to improve model performance — e.g., extracting date parts, combining columns, encoding categories.",
    "classification": "**Classification** predicts a discrete label/category (e.g., spam or not spam). Algorithms: Logistic Regression, Random Forest, Decision Tree, SVM, KNN.",
    "regression": "**Regression** predicts a continuous number (e.g., house price). Algorithms: Linear Regression, Random Forest Regressor, Decision Tree Regressor.",
    "cross validation": "**Cross-Validation** (k-fold) splits data into k parts, trains on k-1 parts and tests on the remaining one, repeating k times. Gives a more robust performance estimate.",
    "normalisation": "**Normalisation** scales features to [0,1] range. Useful for distance-based algorithms like KNN or when features have very different scales.",
    "standardisation": "**Standardisation** transforms features to have mean=0 and std=1. Preferred for algorithms that assume Gaussian distributions like Logistic Regression.",
    "confusion matrix": "**Confusion Matrix** shows TP, TN, FP, FN counts. From it you derive Precision, Recall, F1 Score — key for classification evaluation.",
    "hyperparameter": "**Hyperparameters** are settings you choose before training (e.g., `n_estimators` in Random Forest). Tune them using GridSearchCV or RandomizedSearchCV.",
    "csv": "To upload a CSV: use the sidebar file uploader. Ensure your CSV has a header row, no blank column names, and the target column contains the values you want to predict.",
    "upload": "Click **Browse files** in the sidebar to upload your CSV dataset. After uploading, select the **target column** in the Dataset tab.",
    "target": "The **target column** is the column you want the model to predict. Select it from the dropdown in the Dataset tab after uploading your CSV.",
    "feasibility": "Feasibility is assessed from the best model score: **> 85%** = Highly Feasible, **65–85%** = Moderate, **< 65%** = Low — meaning the model may need more data or better features.",
    "pdf": "After running the analysis, a **Download Report** button appears. Click it to get a PDF with the best model, score, feasibility, and visualisation charts.",
    "voice": "After analysis, an **audio summary** is auto-generated using gTTS. You can play it directly in the app — it reads out the best model, score, and feasibility.",
    "heatmap": "The **Correlation Heatmap** shows pairwise correlations between numeric features. Values near +1 or -1 indicate strong relationships; near 0 means little correlation.",
}

def chatbot_reply(user_msg: str, summary: dict) -> str:
    msg = user_msg.lower().strip()

    # ── Greetings
    greets = ["hi", "hello", "hey", "hii", "helo", "good morning", "good evening"]
    if any(msg == g or msg.startswith(g + " ") for g in greets):
        return f"👋 Hello! I'm your ML AI Advisor assistant. Ask me anything about machine learning, how to use this app, or your latest analysis results!"

    # ── Analysis-aware answers
    if summary:
        if any(w in msg for w in ["best model", "which model", "top model", "winner"]):
            return f"🏆 Your best model was **{summary.get('best_model','N/A')}** with a score of **{summary.get('score',0)*100:.1f}%**."
        if any(w in msg for w in ["score", "accuracy", "result", "r2", "performance"]):
            return f"📊 The best model scored **{summary.get('score',0)*100:.1f}%**. Problem type: **{summary.get('problem','N/A')}**."
        if any(w in msg for w in ["feasib", "good", "reliable", "trust"]):
            return f"✅ Feasibility: **{summary.get('feasibility','N/A')}**. {'The model is production-ready!' if summary.get('feasibility')=='Highly Feasible' else 'Consider collecting more data or engineering better features.'}"

    # ── Knowledge base lookup
    for keyword, answer in ML_KB.items():
        if keyword in msg:
            return answer

    # ── Catch-all
    return (
        "🤔 I'm not sure about that specific question. Try asking about:\n"
        "- A specific ML algorithm (e.g. *'What is Random Forest?'*)\n"
        "- Metrics (e.g. *'Explain R2 score'*)\n"
        "- App usage (e.g. *'How do I upload a CSV?'*)\n"
        "- Your results (e.g. *'What was my best model?'*)"
    )


def show_chatbot():
    st.markdown("### 🤖 ML Doubt Clarification Chatbot")
    st.caption("Ask anything about machine learning concepts, this app, or your analysis results.")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="bubble-bot">
            👋 Hi! I'm your ML AI Advisor chatbot. Ask me about:<br>
            • Machine learning algorithms & concepts<br>
            • Your analysis results<br>
            • How to use this app<br>
            • Metrics like R², Accuracy, Feasibility
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-name-user">You</div><div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-name-bot">🤖 Advisor</div><div class="bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Your question",
            key="chat_input",
            placeholder="e.g. What is Random Forest? / Why did my score drop?",
            label_visibility="collapsed"
        )
    with col2:
        send = st.button("Send ➤", use_container_width=True)

    if send and user_input.strip():
        reply = chatbot_reply(user_input, st.session_state.analysis_summary)
        st.session_state.chat_history.append({"role": "user",    "content": user_input})
        st.session_state.chat_history.append({"role": "assistant","content": reply})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()

    # Quick suggestion buttons
    st.markdown("**💡 Quick questions:**")
    suggestions = [
        "What is Random Forest?",
        "Explain R2 score",
        "What is overfitting?",
        "How do I upload a CSV?",
        "What was my best model?",
        "What does feasibility mean?"
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                reply = chatbot_reply(s, st.session_state.analysis_summary)
                st.session_state.chat_history.append({"role": "user",     "content": s})
                st.session_state.chat_history.append({"role": "assistant","content": reply})
                st.rerun()


# ═══════════════════════════════════════════
#  MAIN APP  (post-login)
# ═══════════════════════════════════════════
def show_main_app():

    # ── Header
    st.markdown(f"""
    <div class="app-header">
      <div>
        <h1>🚀 ML AI Advisor</h1>
        <p>Automated Machine Learning · Analysis · Insights</p>
      </div>
      <div class="user-chip">✅ {st.session_state.full_name} (@{st.session_state.username})</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar
    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown(f"👤 **{st.session_state.full_name}**")
    st.sidebar.markdown(f"`@{st.session_state.username}`")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("📌 Navigation", ["🏠 Home", "💬 Chatbot", "ℹ️ About"])

    if page == "🏠 Home":
        file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
        st.sidebar.markdown("---")

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in", "username", "full_name", "chat_history",
                  "analysis_done", "analysis_summary"]:
            st.session_state[k] = False if k == "logged_in" else "" if k in ["username","full_name"] else [] if k == "chat_history" else {} if k == "analysis_summary" else False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Nithin Mathew | ML AI Advisor")

    # ══════════════════════════════════════
    #  PAGE: HOME
    # ══════════════════════════════════════
    if page == "🏠 Home":

        if not file:
            st.info("👆 Upload a CSV file from the sidebar to begin your analysis.")
            # Show feature cards
            c1, c2, c3, c4 = st.columns(4)
            for col, icon, title, desc in [
                (c1,"🤖","Auto ML","Detects regression or classification automatically"),
                (c2,"📊","Visualisations","Heatmaps and distribution plots"),
                (c3,"🔊","Voice Report","Audio summary of your results"),
                (c4,"📄","PDF Export","Download a full analysis report"),
            ]:
                with col:
                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;text-align:center">
                    <div style="font-size:2rem">{icon}</div>
                    <div style="color:#58a6ff;font-weight:700;margin:8px 0 4px">{title}</div>
                    <div style="color:#8b949e;font-size:0.82rem">{desc}</div>
                    </div>""", unsafe_allow_html=True)
            return

        df = pd.read_csv(file)
        tab1, tab2, tab3 = st.tabs(["📂 Dataset", "🤖 Analysis", "📊 Visuals"])

        # ── Tab 1: Dataset
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows",    df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing", int(df.isnull().sum().sum()))
            target = st.selectbox("🎯 Target Column", df.columns)
            st.session_state["target_col"] = target

        # ── Tab 2: Analysis
        with tab2:
            target = st.session_state.get("target_col", df.columns[0])
            X = pd.get_dummies(df.drop(columns=[target]))
            y = df[target]
            X = X.fillna(X.mean(numeric_only=True))
            try:
                y = pd.to_numeric(y)
            except Exception:
                pass

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            problem = "classification" if y.nunique() < 10 else "regression"
            st.info(f"🔍 Detected Problem Type: **{problem.title()}**")

            if st.button("🚀 Run Analysis", use_container_width=True):
                results = []
                trained_models = {}

                with st.spinner("Training models…"):
                    if problem == "regression":
                        models = {
                            "Linear Regression":  LinearRegression(),
                            "Random Forest":      RandomForestRegressor(n_estimators=100),
                            "Decision Tree":      DecisionTreeRegressor()
                        }
                        for n, m in models.items():
                            m.fit(X_train, y_train)
                            pred = m.predict(X_test)
                            sc = r2_score(y_test, pred)
                            results.append({"Model": n, "R² Score": round(sc, 4), "Score %": f"{sc*100:.2f}%"})
                            trained_models[n] = sc
                    else:
                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=2000),
                            "Random Forest":       RandomForestClassifier(n_estimators=100),
                            "Decision Tree":       DecisionTreeClassifier()
                        }
                        for n, m in models.items():
                            m.fit(X_train, y_train)
                            pred = m.predict(X_test)
                            sc = accuracy_score(y_test, pred)
                            results.append({"Model": n, "Accuracy": round(sc, 4), "Score %": f"{sc*100:.2f}%"})
                            trained_models[n] = sc

                results_df = pd.DataFrame(results)
                best_name  = max(trained_models, key=trained_models.get)
                best_score = trained_models[best_name]

                feasibility = (
                    "Highly Feasible" if best_score > 0.85
                    else "Moderate"    if best_score > 0.65
                    else "Low"
                )

                # Store for chatbot
                st.session_state.analysis_summary = {
                    "best_model":  best_name,
                    "score":       best_score,
                    "problem":     problem,
                    "feasibility": feasibility
                }
                st.session_state.analysis_done = True

                # ── Display
                st.dataframe(results_df, use_container_width=True)
                col1, col2, col3 = st.columns(3)
                col1.success(f"🏆 Best Model: **{best_name}**")
                col2.metric("Score", f"{best_score*100:.2f}%")
                col3.metric("Feasibility", feasibility)

                # ── AI Advisor
                st.subheader("🤖 AI Advisor Insight")
                if "Forest" in best_name:
                    advice = "✅ Random Forest captured complex non-linear patterns in your data. This suggests your features have intricate relationships. Consider feature importance analysis to optimise."
                elif "Linear" in best_name:
                    advice = "✅ A Linear model performed best — your data has strong linear relationships. This model is fast, interpretable, and production-ready."
                else:
                    advice = "⚠️ Decision Tree performed best — consider using Random Forest for more robustness, as Decision Trees can overfit."
                st.info(advice)

                # ── Charts
                heatmap_path = "heatmap.png"
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('#161b22')
                ax.set_facecolor('#161b22')
                sns.heatmap(df.corr(numeric_only=True), ax=ax,
                            cmap="Blues", annot=True, fmt=".1f",
                            linewidths=0.5, linecolor='#30363d',
                            annot_kws={"color":"#e6edf3","size":8})
                ax.tick_params(colors='#8b949e')
                plt.tight_layout()
                plt.savefig(heatmap_path, facecolor='#161b22', bbox_inches='tight')
                plt.close()

                dist_path = "dist.png"
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                fig2.patch.set_facecolor('#161b22')
                ax2.set_facecolor('#0d1117')
                sns.histplot(df[target], ax=ax2, color='#1f6feb', edgecolor='#58a6ff', bins=30)
                ax2.set_title(f"Distribution of {target}", color='#e6edf3')
                ax2.tick_params(colors='#8b949e')
                ax2.spines[:].set_color('#30363d')
                plt.tight_layout()
                plt.savefig(dist_path, facecolor='#161b22', bbox_inches='tight')
                plt.close()

                # ── Voice
                try:
                    text = (f"Analysis complete. Best model is {best_name}. "
                            f"Score is {best_score*100:.1f} percent. "
                            f"Feasibility is {feasibility}.")
                    tts = gTTS(text)
                    tts.save("voice.mp3")
                    st.audio("voice.mp3")
                except Exception:
                    st.caption("Voice output unavailable (network issue).")

                # ── PDF
                try:
                    pdf_path = "ML_Report.pdf"
                    doc = SimpleDocTemplate(pdf_path)
                    styles = getSampleStyleSheet()
                    content = [
                        Paragraph("ML AI Advisor — Analysis Report", styles['Title']),
                        Spacer(1, 10),
                        Paragraph(f"Analyst: {st.session_state.full_name}", styles['Normal']),
                        Paragraph(f"Problem Type: {problem.title()}", styles['Normal']),
                        Paragraph(f"Best Model: {best_name}", styles['Normal']),
                        Paragraph(f"Score: {best_score*100:.2f}%", styles['Normal']),
                        Paragraph(f"Feasibility: {feasibility}", styles['Normal']),
                        Spacer(1, 12),
                        Paragraph("AI Advisor Insight:", styles['Heading2']),
                        Paragraph(advice, styles['Normal']),
                        Spacer(1, 12),
                        Paragraph("Correlation Heatmap", styles['Heading2']),
                        Image(heatmap_path, width=420, height=280),
                        Spacer(1, 8),
                        Paragraph(f"Target Distribution — {target}", styles['Heading2']),
                        Image(dist_path, width=420, height=240),
                    ]
                    doc.build(content)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=f.read(),
                            file_name="ML_AI_Advisor_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                except Exception as e:
                    st.warning(f"PDF generation failed: {e}")

        # ── Tab 3: Visuals
        with tab3:
            st.subheader("📊 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            sns.heatmap(df.corr(numeric_only=True), ax=ax,
                        cmap="Blues", annot=True, fmt=".1f",
                        linewidths=0.5, linecolor='#30363d',
                        annot_kws={"color":"#e6edf3","size":8})
            ax.tick_params(colors='#8b949e')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.subheader("📈 Numeric Distributions")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                sel = st.selectbox("Select column", num_cols)
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                fig2.patch.set_facecolor('#161b22')
                ax2.set_facecolor('#0d1117')
                sns.histplot(df[sel], ax=ax2, color='#1f6feb', edgecolor='#58a6ff', bins=30)
                ax2.set_title(f"Distribution of {sel}", color='#e6edf3')
                ax2.tick_params(colors='#8b949e')
                ax2.spines[:].set_color('#30363d')
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)

    # ══════════════════════════════════════
    #  PAGE: CHATBOT
    # ══════════════════════════════════════
    elif page == "💬 Chatbot":
        show_chatbot()

    # ══════════════════════════════════════
    #  PAGE: ABOUT
    # ══════════════════════════════════════
    elif page == "ℹ️ About":
        st.markdown("""
        ## 🚀 Project Overview
        This application is an intelligent Machine Learning assistant that:
        - Automatically detects problem type (Regression / Classification)
        - Trains multiple ML models and selects the best one
        - Generates insights, voice summaries, and downloadable PDF reports
        - Includes a chatbot for ML doubt clarification

        ## 🧠 New Features (v2.0)
        | Feature | Description |
        |---|---|
        | 🔐 Login / Sign Up | Secure user authentication with hashed passwords |
        | 💬 ML Chatbot | Ask questions about ML concepts, app usage, or your results |
        | 🎨 Dark Theme UI | Fully redesigned GitHub-inspired dark interface |
        | 📊 Themed Charts | All charts styled to match the dark UI |

        ## ⚙️ Technologies Used
        - **Python** 🐍 · **Streamlit** 🌐 · **Scikit-learn** 🤖
        - **Matplotlib & Seaborn** 📊 · **gTTS** 🔊 · **ReportLab** 📄

        ## 👨‍💻 Developer
        Built by **Nithin Mathew** 💡  
        AIML Student | Future AI Engineer 🚀  
        © 2026 Nithin Mathew
        """)


# ═══════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════
if not st.session_state.logged_in:
    # centre logo above auth form
    st.markdown("""
    <div style='text-align:center;padding:30px 0 10px'>
      <span style='font-size:3rem'>🚀</span>
      <h1 style='color:#58a6ff;margin:6px 0 2px'>ML AI Advisor</h1>
      <p style='color:#8b949e;font-size:0.9rem'>by Nithin Mathew &nbsp;|&nbsp; Automated Machine Learning Platform</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.auth_page == "login":
        show_login()
    else:
        show_signup()
else:
    show_main_app()
