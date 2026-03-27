# ======================================================
# 🚀 ML AI ADVISOR v3.0 — FULL FINAL (GEMINI VERSION)
# Auth + ML + Chatbot + Gemini AI (NO CLAUDE)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib, json, os, uuid
import google.generativeai as genai

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ML AI Advisor", layout="wide")

USERS_FILE = "users.json"

# -----------------------------
# AUTH SYSTEM
# -----------------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f)


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def register(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = hash_pw(password)
    save_users(users)
    return True


def login(username, password):
    users = load_users()
    return username in users and users[username] == hash_pw(password)

# -----------------------------
# GEMINI AI
# -----------------------------
def call_gemini(messages, api_key):
    if not api_key:
        return "⚠️ Add Google API key in sidebar"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = "You are an expert ML assistant.\n\n"

        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            prompt += f"{role}: {m['content']}\n"

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# SESSION STATE
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "mode" not in st.session_state:
    st.session_state.mode = "login"

# -----------------------------
# LOGIN / SIGNUP
# -----------------------------
if not st.session_state.logged_in:

    st.title("🔐 Authentication")

    col1, col2 = st.columns(2)

    if col1.button("Login Mode"):
        st.session_state.mode = "login"

    if col2.button("Signup Mode"):
        st.session_state.mode = "signup"

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.session_state.mode == "login":
        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Logged in 🚀")
                st.rerun()
            else:
                st.error("Invalid credentials")

    else:
        if st.button("Signup"):
            if register(username, password):
                st.success("Account created! Switch to login")
                st.session_state.mode = "login"
            else:
                st.error("User already exists")

    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title(f"👤 {st.session_state.user}")

api_input = st.sidebar.text_input("Google Gemini API Key", type="password")

if st.sidebar.button("Save Key"):
    st.session_state.api_key = api_input
    st.sidebar.success("Saved")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -----------------------------
# MAIN APP
# -----------------------------
st.title("🚀 ML AI Advisor (Gemini)")

# -----------------------------
# DATA ANALYSIS
# -----------------------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    target = st.selectbox("Select Target", df.columns)

    if st.button("Run ML"):
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]

        try:
            y = pd.to_numeric(y)
        except:
            pass

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        problem = "classification" if y.nunique() < 10 else "regression"

        results = {}

        if problem == "regression":
            models = {
                "Linear": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))
                results[name] = score

        else:
            models = {
                "Logistic": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                results[name] = score

        best_model = max(results, key=results.get)
        best_score = results[best_model]

        st.success(f"🏆 Best Model: {best_model} ({best_score*100:.2f}%)")

        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), ax=ax)
        st.pyplot(fig)

# -----------------------------
# CHATBOT
# -----------------------------
st.header("🤖 AI Chatbot")

for m in st.session_state.chat:
    st.write(f"{m['role']}: {m['content']}")

user_input = st.text_input("Ask anything...")

if st.button("Send") and user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

    reply = call_gemini(st.session_state.chat, st.session_state.api_key)

    st.session_state.chat.append({"role": "assistant", "content": reply})

    st.rerun()
