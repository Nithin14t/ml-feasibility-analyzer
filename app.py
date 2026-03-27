# ======================================================
# 🚀 ML AI ADVISOR v4.0 — ENTERPRISE VERSION
# Gemini + Memory + Auth + Collaboration + PDF + Voice
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, os, hashlib, uuid
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

# -----------------------------
# FILES
# -----------------------------
MEMORY_FILE = "memory.json"
USERS_FILE = "users.json"
SESSION_FILE = "sessions.json"

st.set_page_config(page_title="ML AI Advisor", layout="wide")

# -----------------------------
# BASIC STORAGE
# -----------------------------
def load_file(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_file(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# AUTH SYSTEM 🔐
# -----------------------------
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def register(username, password):
    users = load_file(USERS_FILE)
    if username in users:
        return False
    users[username] = hash_pw(password)
    save_file(USERS_FILE, users)
    return True


def login(username, password):
    users = load_file(USERS_FILE)
    return username in users and users[username] == hash_pw(password)

# -----------------------------
# MEMORY 🧠
# -----------------------------
def load_memory(): return load_file(MEMORY_FILE)

def save_memory(mem): save_file(MEMORY_FILE, mem)


def update_memory(user, u, a):
    mem = load_memory()
    mem.setdefault(user, []).append({"u": u, "a": a})
    mem[user] = mem[user][-20:]
    save_memory(mem)


def get_memory(user):
    mem = load_memory()
    ctx = ""
    for m in mem.get(user, [])[-10:]:
        ctx += f"User:{m['u']}\nAI:{m['a']}\n"
    return ctx

# -----------------------------
# GEMINI 🤖
# -----------------------------
def init_gemini(key):
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-1.5-flash")


def ask_ai(user, history, key):
    if not key: return "Add API key"
    model = init_gemini(key)

    prompt = "You are ML expert AI.\n" + get_memory(user)

    for m in history:
        role = "User" if m["role"] == "user" else "AI"
        prompt += f"{role}:{m['content']}\n"

    return model.generate_content(prompt).text

# -----------------------------
# SESSION STATE
# -----------------------------
if "login" not in st.session_state:
    st.session_state.login = False
if "user" not in st.session_state:
    st.session_state.user = ""
if "chat" not in st.session_state:
    st.session_state.chat = []
if "key" not in st.session_state:
    st.session_state.key = ""

# -----------------------------
# LOGIN UI
# -----------------------------
if not st.session_state.login:
    st.title("🔐 Login / Signup")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    if col1.button("Login"):
        if login(u, p):
            st.session_state.login = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid")

    if col2.button("Signup"):
        if register(u, p):
            st.success("Created")
        else:
            st.error("User exists")

    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title(f"👤 {st.session_state.user}")

key = st.sidebar.text_input("Gemini Key", type="password")
if st.sidebar.button("Save Key"):
    st.session_state.key = key

if st.sidebar.button("Logout"):
    st.session_state.login = False
    st.rerun()

# -----------------------------
# MAIN APP
# -----------------------------
st.title("🚀 ML AI Advisor Enterprise")

# -----------------------------
# DATA + ML
# -----------------------------
file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    target = st.selectbox("Target", df.columns)

    if st.button("Run ML"):
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        models = {
            "RF": RandomForestRegressor() if y.nunique()>10 else RandomForestClassifier(),
            "DT": DecisionTreeRegressor() if y.nunique()>10 else DecisionTreeClassifier()
        }

        scores = {}
        for n,m in models.items():
            m.fit(X_train,y_train)
            pred = m.predict(X_test)
            scores[n] = r2_score(y_test,pred) if y.nunique()>10 else accuracy_score(y_test,pred)

        best = max(scores,key=scores.get)
        st.success(f"Best: {best} {scores[best]*100:.2f}%")

        # PDF
        pdf = "report.pdf"
        doc = SimpleDocTemplate(pdf)
        styles = getSampleStyleSheet()
        doc.build([Paragraph(f"Best Model: {best}", styles['Normal'])])

        with open(pdf,"rb") as f:
            st.download_button("Download Report",f,"report.pdf")

        # Voice
        tts = gTTS(f"Best model is {best}")
        tts.save("voice.mp3")
        st.audio("voice.mp3")

# -----------------------------
# CHATBOT
# -----------------------------
st.header("🤖 AI Chat")

for m in st.session_state.chat:
    st.write(f"{m['role']}: {m['content']}")

msg = st.text_input("Ask...")

if st.button("Send") and msg:
    st.session_state.chat.append({"role":"user","content":msg})

    reply = ask_ai(st.session_state.user, st.session_state.chat, st.session_state.key)

    st.session_state.chat.append({"role":"ai","content":reply})

    update_memory(st.session_state.user, msg, reply)

    st.rerun()

# -----------------------------
# COLLAB (simple)
# -----------------------------
st.header("👥 Collaboration")

sessions = load_file(SESSION_FILE)

sid = st.text_input("Session ID")

if st.button("Create Session"):
    sid = str(uuid.uuid4())[:6]
    sessions[sid] = []
    save_file(SESSION_FILE, sessions)
    st.success(f"Session: {sid}")

if st.button("Join") and sid in sessions:
    st.success("Joined")

msg2 = st.text_input("Team message")
if st.button("Send to team") and sid in sessions:
    sessions[sid].append(f"{st.session_state.user}:{msg2}")
    save_file(SESSION_FILE, sessions)

if sid in sessions:
    for m in sessions[sid]:
        st.write(m)
