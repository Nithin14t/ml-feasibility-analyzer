"""
╔══════════════════════════════════════════════════════════════════╗
║          ML AI ADVISOR  v3.0  —  by Nithin Mathew               ║
║ 
║  • Real-time Collaboration Workspace                             ║
║  • Multi-user Auth  •  Live Comments  •  Shared Sessions         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib, json, os, time, uuid, requests
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ML AI Advisor v3",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

USERS_FILE     = "users.json"
COLLAB_FILE    = "collab_sessions.json"
COMMENTS_FILE  = "comments.json"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"

# ──────────────────────────────────────────────
#  PERSISTENCE HELPERS
# ──────────────────────────────────────────────
def _load(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_users():     return _load(USERS_FILE, {})
def save_users(d):    _save(USERS_FILE, d)
def load_collab():    return _load(COLLAB_FILE, {})
def save_collab(d):   _save(COLLAB_FILE, d)
def load_comments():  return _load(COMMENTS_FILE, {})
def save_comments(d): _save(COMMENTS_FILE, d)

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password, email, full_name, role="Analyst"):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password": hash_pw(password), "email": email,
        "full_name": full_name, "role": role,
        "joined": datetime.now().strftime("%Y-%m-%d"),
        "avatar_color": "#" + hashlib.md5(username.encode()).hexdigest()[:6]
    }
    save_users(users)
    return True, "Account created!"

def authenticate(username, password):
    users = load_users()
    if username not in users:
        return False, "User not found. Please sign up."
    if users[username]["password"] != hash_pw(password):
        return False, "Wrong password."
    return True, users[username]

# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────
DEFAULTS = {
    "logged_in": False, "username": "", "full_name": "",
    "user_info": {}, "auth_page": "login",
    "chat_history": [],
    "analysis_summary": {},
    "collab_session_id": None,
    "anthropic_key": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117; font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"]   { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

.top-bar {
    background: linear-gradient(90deg,#1a1f2e,#161b22);
    border: 1px solid #21262d; border-radius: 12px;
    padding: 16px 24px; display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 20px;
}
.top-bar-title { color:#58a6ff; font-size:1.35rem; font-weight:700; }
.top-bar-sub   { color:#8b949e; font-size:0.8rem; margin-top:2px; }
.user-pill {
    display:flex; align-items:center; gap:8px;
    background:#21262d; border:1px solid #30363d;
    border-radius:20px; padding:6px 14px;
    font-size:0.82rem; color:#3fb950; font-weight:600;
}
.user-avatar {
    width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:0.75rem; color:#fff;
}
.auth-wrap {
    max-width:420px; margin:48px auto 0;
    background:#161b22; border:1px solid #30363d;
    border-radius:16px; padding:36px 40px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}
.auth-logo  { text-align:center; font-size:2.8rem; margin-bottom:6px; }
.auth-title { text-align:center; color:#58a6ff; font-size:1.5rem; font-weight:700; }
.auth-sub   { text-align:center; color:#8b949e; font-size:0.85rem; margin-bottom:24px; }
.msg-user {
    background:#1f6feb; color:#fff;
    border-radius:16px 16px 4px 16px;
    padding:10px 15px; margin:4px 0 4px 48px;
    font-size:0.88rem; line-height:1.55;
}
.msg-bot {
    background:#21262d; color:#e6edf3;
    border:1px solid #30363d;
    border-radius:16px 16px 16px 4px;
    padding:10px 15px; margin:4px 48px 4px 0;
    font-size:0.88rem; line-height:1.55;
}
.msg-collab {
    background:#1c2128; color:#e6edf3;
    border:1px solid #30363d; border-radius:12px;
    padding:10px 14px; margin:6px 0; font-size:0.87rem;
}
.msg-label-user  { text-align:right; font-size:0.68rem; color:#8b949e; margin-bottom:2px; }
.msg-label-bot   { font-size:0.68rem; color:#58a6ff; margin-bottom:2px; }
.msg-label-collab{ font-size:0.72rem; color:#8b949e; margin-bottom:3px; }
.collab-name     { font-weight:600; }
.stat-card {
    background:#161b22; border:1px solid #21262d;
    border-radius:12px; padding:18px 20px; text-align:center;
}
.stat-val { color:#3fb950; font-size:1.6rem; font-weight:700; }
.stat-lbl { color:#8b949e; font-size:0.75rem; margin-top:2px; }
.collab-card {
    background:#161b22; border:1px solid #21262d;
    border-radius:12px; padding:16px 20px; margin:8px 0;
}
.collab-card-title { color:#58a6ff; font-weight:600; font-size:0.95rem; }
.collab-card-sub   { color:#8b949e; font-size:0.8rem; margin-top:3px; }
.member-chip {
    display:inline-flex; align-items:center; gap:6px;
    background:#21262d; border:1px solid #30363d;
    border-radius:100px; padding:4px 12px;
    font-size:0.78rem; color:#e6edf3; margin:3px;
}
.online-dot {
    width:7px; height:7px; background:#3fb950; border-radius:50%;
}
.ai-insight {
    background:linear-gradient(135deg,#1a2744,#162032);
    border:1px solid #1f4080; border-left:3px solid #58a6ff;
    border-radius:10px; padding:16px 18px; margin:12px 0;
    color:#c9d1d9; font-size:0.9rem; line-height:1.7;
}
.ai-insight-title { color:#58a6ff; font-weight:700; margin-bottom:8px; font-size:1rem; }
.session-badge {
    background:#1a3a1a; border:1px solid #238636;
    border-radius:8px; padding:8px 12px;
    font-size:0.78rem; color:#3fb950; margin:8px 0;
}
.comment-block {
    background:#161b22; border:1px solid #21262d;
    border-radius:10px; padding:12px 16px; margin:6px 0;
}
.comment-author { color:#58a6ff; font-size:0.8rem; font-weight:600; }
.comment-text   { color:#c9d1d9; font-size:0.88rem; margin-top:4px; }
.comment-time   { color:#484f58; font-size:0.7rem; margin-top:4px; }
[data-baseweb="tab"]   { color:#8b949e !important; font-size:0.88rem !important; }
[aria-selected="true"] { color:#58a6ff !important; border-bottom:2px solid #58a6ff !important; }
[data-testid="stMetric"] label { color:#8b949e !important; }
[data-testid="stMetricValue"]  { color:#3fb950 !important; font-weight:700; }
input, textarea {
    background:#0d1117 !important; color:#e6edf3 !important;
    border:1px solid #30363d !important; border-radius:8px !important;
}
.stButton > button {
    background:#238636; color:#fff; border:none;
    border-radius:8px; font-weight:600; font-size:0.88rem; transition:all .2s;
}
.stButton > button:hover { background:#2ea043; transform:translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  GENERATIVE AI  —  Claude API
# ══════════════════════════════════════════════
def call_claude(messages: list, system_prompt: str, api_key: str) -> str:
    if not api_key:
        return (
            "⚠️ **No API key set.** Please enter your Anthropic API key in the "
            "sidebar under ⚙️ Settings to enable the Generative AI chatbot.\n\n"
            "Get a free key at [console.anthropic.com](https://console.anthropic.com)"
        )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": messages
    }
    try:
        r = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 401: return "❌ Invalid API key. Check your key in Settings."
        if code == 429: return "⏳ Rate limit hit. Wait a moment and retry."
        return f"❌ API Error {code}: {e.response.text[:200]}"
    except Exception as e:
        return f"❌ Connection error: {str(e)}"


def build_system_prompt(summary: dict, username: str) -> str:
    ctx = ""
    if summary:
        ctx = f"""
The user has completed an ML analysis:
- Problem Type: {summary.get('problem','unknown')}
- Best Model: {summary.get('best_model','unknown')}
- Score: {summary.get('score',0)*100:.1f}%
- Feasibility: {summary.get('feasibility','unknown')}
- All model scores: {summary.get('all_scores',{})}
- Dataset: {summary.get('n_rows','?')} rows x {summary.get('n_cols','?')} columns
- Target: {summary.get('target','?')}
- Features (first 10): {summary.get('features',[])}
Use this context for personalised advice.
"""
    return f"""You are the ML AI Advisor — an expert machine learning assistant.
The user is: {username}.

Your job:
1. Answer ML/AI/data science questions clearly and thoroughly
2. Explain concepts with examples and intuition
3. Give personalised feedback based on the user's analysis when relevant
4. Suggest concrete next steps (hyperparameter tuning, feature engineering, etc.)
5. Be encouraging — the user is an AIML student
6. Use markdown: **bold**, bullets, code blocks

{ctx}

Tone: Friendly, expert, concise but complete. Never give one-word answers.
If asked something off-topic, gently redirect to ML/data science."""


# ══════════════════════════════════════════════
#  COLLABORATION HELPERS
# ══════════════════════════════════════════════
def create_session(owner: str, name: str) -> str:
    sessions = load_collab()
    sid = str(uuid.uuid4())[:8].upper()
    sessions[sid] = {
        "name": name, "owner": owner, "members": [owner],
        "messages": [], "shared_results": None,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    save_collab(sessions)
    return sid

def join_session(username: str, sid: str):
    sessions = load_collab()
    if sid not in sessions:
        return False, "Session not found."
    if username not in sessions[sid]["members"]:
        sessions[sid]["members"].append(username)
        save_collab(sessions)
    return True, sessions[sid]

def post_collab_msg(sid, user, msg, msg_type="text"):
    sessions = load_collab()
    if sid not in sessions: return
    sessions[sid]["messages"].append({
        "user": user, "msg": msg, "type": msg_type,
        "time": datetime.now().strftime("%H:%M")
    })
    save_collab(sessions)

def get_collab_messages(sid):
    return load_collab().get(sid, {}).get("messages", [])

def push_shared_results(sid, results):
    sessions = load_collab()
    if sid in sessions:
        sessions[sid]["shared_results"] = results
        save_collab(sessions)

def get_shared_results(sid):
    return load_collab().get(sid, {}).get("shared_results")

def post_comment(sid, username, text):
    comments = load_comments()
    if sid not in comments: comments[sid] = []
    comments[sid].append({
        "user": username, "text": text,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    save_comments(comments)

def get_comments(sid):
    return load_comments().get(sid, [])


# ══════════════════════════════════════════════
#  AUTH PAGES
# ══════════════════════════════════════════════
def page_login():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div class="auth-wrap">
          <div class="auth-logo">🚀</div>
          <div class="auth-title">ML AI Advisor</div>
          <div class="auth-sub">Sign in to continue</div>
        </div>""", unsafe_allow_html=True)
        username = st.text_input("👤 Username", placeholder="your_username", key="li_u")
        password = st.text_input("🔒 Password", type="password", placeholder="••••••••", key="li_p")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Login", use_container_width=True):
                if not username or not password:
                    st.error("Fill all fields.")
                else:
                    ok, result = authenticate(username, password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = username
                        st.session_state.full_name = result["full_name"]
                        st.session_state.user_info = result
                        st.success(f"Welcome, {result['full_name']}! 🎉")
                        time.sleep(0.6); st.rerun()
                    else:
                        st.error(result)
        with c2:
            if st.button("📝 New? Sign Up", use_container_width=True):
                st.session_state.auth_page = "signup"; st.rerun()
        st.caption("New here? Click **Sign Up** above.")


def page_signup():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div class="auth-wrap">
          <div class="auth-logo">✨</div>
          <div class="auth-title">Create Account</div>
          <div class="auth-sub">Join ML AI Advisor for free</div>
        </div>""", unsafe_allow_html=True)
        full_name = st.text_input("👤 Full Name",  placeholder="Nithin Mathew", key="su_n")
        email     = st.text_input("📧 Email",      placeholder="you@email.com", key="su_e")
        username  = st.text_input("🆔 Username",   placeholder="nithin_ml",     key="su_u")
        role      = st.selectbox("🎓 Role", ["Student","Analyst","Data Scientist","Researcher","Engineer"])
        c1, c2 = st.columns(2)
        with c1: password = st.text_input("🔒 Password", type="password", key="su_p")
        with c2: confirm  = st.text_input("🔒 Confirm",  type="password", key="su_c")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ Create Account", use_container_width=True):
                if not all([full_name, email, username, password, confirm]):
                    st.error("Fill all fields.")
                elif len(password) < 6:
                    st.error("Password needs 6+ chars.")
                elif password != confirm:
                    st.error("Passwords don't match.")
                else:
                    ok, msg = register_user(username, password, email, full_name, role)
                    if ok:
                        st.success(msg + " Redirecting…")
                        time.sleep(1)
                        st.session_state.auth_page = "login"; st.rerun()
                    else:
                        st.error(msg)
        with b2:
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.auth_page = "login"; st.rerun()


# ══════════════════════════════════════════════
#  GENERATIVE AI CHATBOT PAGE
# ══════════════════════════════════════════════
def page_chatbot():
    st.markdown("### 🤖 Generative AI Assistant")
    st.caption("Powered by **Claude claude-sonnet-4-20250514** (Anthropic) — real language model, not if-else rules.")

    api_key = st.session_state.anthropic_key
    if not api_key:
        st.warning("⚠️ Enter your Anthropic API key in **⚙️ Settings** (sidebar) to activate the AI.")

    # Display chat
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="msg-bot">
        👋 Hi! I'm your <strong>Generative AI ML Advisor</strong>, powered by Claude.<br><br>
        I can help you with:<br>
        • Deep explanations of any ML algorithm or concept<br>
        • Personalised advice based on <em>your</em> analysis results<br>
        • Feature engineering & model improvement strategies<br>
        • Python code snippets and formula derivations<br>
        • Career advice for AIML students<br><br>
        <em>Add your Anthropic API key in Settings and start asking!</em>
        </div>""", unsafe_allow_html=True)
    else:
        for m in st.session_state.chat_history:
            if m["role"] == "user":
                st.markdown(f'<div class="msg-label-user">You</div>'
                            f'<div class="msg-user">{m["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-label-bot">🤖 Claude</div>'
                            f'<div class="msg-bot">{m["content"]}</div>', unsafe_allow_html=True)

    # Input
    col1, col2 = st.columns([6, 1])
    with col1:
        user_q = st.text_input("Ask anything…", key="ai_input",
                               placeholder="e.g. Why did Random Forest beat Linear Regression?",
                               label_visibility="collapsed")
    with col2:
        send = st.button("Send ➤", use_container_width=True, key="ai_send")

    if send and user_q.strip():
        with st.spinner("Claude is thinking…"):
            history = st.session_state.chat_history[-20:]
            api_msgs = [{"role": m["role"], "content": m["content"]} for m in history]
            api_msgs.append({"role": "user", "content": user_q})
            system = build_system_prompt(st.session_state.analysis_summary, st.session_state.full_name)
            reply  = call_claude(api_msgs, system, api_key)
        st.session_state.chat_history.append({"role": "user",      "content": user_q})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # Quick prompts
    st.markdown("**💡 Quick prompts:**")
    prompts = [
        "Explain my analysis results in simple terms",
        "Top 3 ways to improve my model score",
        "What is the math behind Random Forest?",
        "How do I handle class imbalance?",
        "Suggest next steps for my ML project",
        "Explain overfitting with a code fix",
        "Bias-variance tradeoff explained",
        "How to do feature importance in Python?",
    ]
    cols = st.columns(4)
    for i, p in enumerate(prompts):
        with cols[i % 4]:
            if st.button(p, key=f"qp_{i}", use_container_width=True):
                with st.spinner("Claude is thinking…"):
                    api_msgs = [{"role": m["role"], "content": m["content"]}
                                for m in st.session_state.chat_history[-20:]]
                    api_msgs.append({"role": "user", "content": p})
                    system = build_system_prompt(st.session_state.analysis_summary, st.session_state.full_name)
                    reply  = call_claude(api_msgs, system, api_key)
                st.session_state.chat_history.append({"role": "user",      "content": p})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []; st.rerun()


# ══════════════════════════════════════════════
#  COLLABORATION PAGE
# ══════════════════════════════════════════════
def page_collaboration():
    st.markdown("### 👥 Collaboration Workspace")
    st.caption("Create or join a session to analyse together, share results, and consult the AI as a team.")

    sid      = st.session_state.collab_session_id
    username = st.session_state.username
    api_key  = st.session_state.anthropic_key

    # ── Not in a session
    if not sid:
        tab_new, tab_join, tab_browse = st.tabs(["➕ New Session", "🔑 Join Session", "📋 Browse"])

        with tab_new:
            sname = st.text_input("Session Name", placeholder="e.g. Iris Classification Sprint")
            if st.button("🚀 Create Session", use_container_width=True):
                if not sname.strip(): st.error("Enter a name.")
                else:
                    new_sid = create_session(username, sname)
                    st.session_state.collab_session_id = new_sid
                    post_collab_msg(new_sid, "System",
                                   f"🎉 Session **{sname}** created by **{username}**", "system")
                    st.success(f"Created! Share ID: **{new_sid}**"); st.rerun()

        with tab_join:
            join_id = st.text_input("Session ID", placeholder="e.g. A3F7B2C1").upper()
            if st.button("🔗 Join", use_container_width=True):
                if not join_id.strip(): st.error("Enter an ID.")
                else:
                    ok, result = join_session(username, join_id)
                    if ok:
                        st.session_state.collab_session_id = join_id
                        post_collab_msg(join_id, "System", f"👤 **{username}** joined.", "system")
                        st.success(f"Joined: **{result['name']}**"); st.rerun()
                    else: st.error(result)

        with tab_browse:
            sessions = load_collab()
            if not sessions: st.info("No sessions yet.")
            for s_id, s_data in sessions.items():
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"""
                    <div class="collab-card">
                      <div class="collab-card-title">📂 {s_data['name']}</div>
                      <div class="collab-card-sub">
                        ID: <code>{s_id}</code> · Owner: {s_data['owner']} ·
                        Members: {len(s_data['members'])} · {s_data['created']}
                      </div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if st.button("Join", key=f"bj_{s_id}", use_container_width=True):
                        ok, result = join_session(username, s_id)
                        if ok:
                            st.session_state.collab_session_id = s_id
                            post_collab_msg(s_id, "System", f"👤 **{username}** joined.", "system")
                            st.rerun()
        return

    # ── Inside session
    sessions = load_collab()
    if sid not in sessions:
        st.session_state.collab_session_id = None; st.rerun()
    sess    = sessions[sid]
    members = sess.get("members", [])

    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="session-badge">
          📂 <strong>{sess['name']}</strong> &nbsp;·&nbsp;
          ID: <code>{sid}</code> &nbsp;·&nbsp; Owner: {sess['owner']}
        </div>""", unsafe_allow_html=True)
        chips = " ".join([
            f'<span class="member-chip"><span class="online-dot"></span>{m}</span>'
            for m in members
        ])
        st.markdown(f"**Members:** {chips}", unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Leave", use_container_width=True):
            st.session_state.collab_session_id = None; st.rerun()

    st.markdown("---")
    t_chat, t_results, t_ai, t_comments = st.tabs([
        "💬 Team Chat", "📊 Shared Results", "🤖 Team AI Consult", "📝 Comments"
    ])

    # ── Team Chat
    with t_chat:
        st.markdown("#### 💬 Live Team Chat")
        msgs = get_collab_messages(sid)
        for m in msgs[-30:]:
            if m.get("type") == "system":
                st.markdown(
                    f"<div style='text-align:center;color:#484f58;font-size:0.78rem;margin:6px 0'>{m['msg']}</div>",
                    unsafe_allow_html=True)
            elif m.get("type") == "ai":
                st.markdown(f"""
                <div class="msg-collab">
                  <div class="msg-label-collab">🤖 <span class="collab-name">Claude</span> · {m['time']}</div>
                  {m['msg']}
                </div>""", unsafe_allow_html=True)
            else:
                you = " (you)" if m["user"] == username else ""
                st.markdown(f"""
                <div class="msg-collab">
                  <div class="msg-label-collab">👤 <span class="collab-name">{m['user']}</span>{you} · {m['time']}</div>
                  {m['msg']}
                </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([5, 1])
        with c1:
            chat_msg = st.text_input("Message…", key="collab_msg",
                                     label_visibility="collapsed",
                                     placeholder="Share insights, tag a result, or ask the team…")
        with c2:
            if st.button("Send 📤", use_container_width=True, key="collab_send"):
                if chat_msg.strip():
                    post_collab_msg(sid, username, chat_msg); st.rerun()

        if st.session_state.analysis_summary:
            if st.button("📊 Share My Analysis to Team", use_container_width=True):
                s = st.session_state.analysis_summary
                post_collab_msg(sid, username,
                    f"📊 **Results from {username}**: Problem={s.get('problem','?').title()}, "
                    f"Best={s.get('best_model','?')}, Score={s.get('score',0)*100:.1f}%, "
                    f"Feasibility={s.get('feasibility','?')}", "system")
                push_shared_results(sid, s)
                st.success("Results shared!"); st.rerun()

    # ── Shared Results
    with t_results:
        st.markdown("#### 📊 Shared Analysis Results")
        shared = get_shared_results(sid)
        if not shared:
            st.info("No results shared yet. Run analysis → Share to team.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="stat-card"><div class="stat-val">{shared.get("best_model","?")}</div><div class="stat-lbl">Best Model</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="stat-val">{shared.get("score",0)*100:.1f}%</div><div class="stat-lbl">Score</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-card"><div class="stat-val">{shared.get("feasibility","?")}</div><div class="stat-lbl">Feasibility</div></div>', unsafe_allow_html=True)

            if shared.get("all_scores"):
                sc_df = pd.DataFrame(list(shared["all_scores"].items()), columns=["Model","Score"])
                sc_df["Score %"] = sc_df["Score"].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(sc_df, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#0d1117')
                colors = ['#238636','#1f6feb','#da3633']
                bars = ax.barh(sc_df["Model"], sc_df["Score"], color=colors[:len(sc_df)])
                ax.set_xlim(0, 1); ax.set_xlabel("Score", color='#8b949e')
                ax.tick_params(colors='#8b949e'); ax.spines[:].set_color('#30363d')
                for bar, val in zip(bars, sc_df["Score"]):
                    ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
                            f'{val*100:.1f}%', va='center', color='#e6edf3', fontsize=9)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    # ── Team AI Consult
    with t_ai:
        st.markdown("#### 🤖 Team AI Consultation")
        st.caption("Ask Claude on behalf of the whole team. Answers are visible to all members.")
        if not api_key:
            st.warning("⚠️ Enter your Anthropic API key in Settings.")

        ai_msgs = [m for m in get_collab_messages(sid) if m.get("type") in ("ai","ai_q")]
        for m in ai_msgs[-10:]:
            if m.get("type") == "ai_q":
                st.markdown(f"""
                <div class="msg-collab">
                  <div class="msg-label-collab">❓ <strong>{m['user']}</strong> asked · {m['time']}</div>
                  {m['msg']}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-insight"><div class="ai-insight-title">🤖 Claude</div>{m["msg"]}</div>',
                            unsafe_allow_html=True)

        c1, c2 = st.columns([5, 1])
        with c1:
            team_q = st.text_input("Ask Claude for the team…", key="team_ai_q",
                                   label_visibility="collapsed",
                                   placeholder="e.g. What model should our team try next?")
        with c2:
            if st.button("Ask 🤖", use_container_width=True, key="team_ai_send"):
                if team_q.strip() and api_key:
                    with st.spinner("Claude consulting…"):
                        shared = get_shared_results(sid) or {}
                        system = build_system_prompt(shared, f"the team '{sess['name']}'")
                        reply  = call_claude([{"role":"user","content":team_q}], system, api_key)
                    post_collab_msg(sid, username, team_q, "ai_q")
                    post_collab_msg(sid, "Claude",  reply,  "ai")
                    st.rerun()

    # ── Comments
    with t_comments:
        st.markdown("#### 📝 Comments & Annotations")
        for c in reversed(get_comments(sid)):
            st.markdown(f"""
            <div class="comment-block">
              <div class="comment-author">👤 {c['user']}</div>
              <div class="comment-text">{c['text']}</div>
              <div class="comment-time">🕐 {c['time']}</div>
            </div>""", unsafe_allow_html=True)
        nc = st.text_area("Add a comment…", key="new_comment",
                          placeholder="e.g. High variance in feature X may explain the low R² score…")
        if st.button("📝 Post Comment", use_container_width=True):
            if nc.strip():
                post_comment(sid, username, nc); st.rerun()


# ══════════════════════════════════════════════
#  HOME (ML ANALYSIS) PAGE
# ══════════════════════════════════════════════
def page_home(file):
    if not file:
        st.markdown("### 👋 Welcome to ML AI Advisor v3.0")
        cols = st.columns(4)
        for col, icon, title, desc in zip(cols, [
            ("🤖","Generative AI","Real Claude AI — not if-else rules"),
            ("👥","Collaboration","Team sessions with shared results"),
            ("📊","Auto ML","Auto-detects regression vs classification"),
            ("📄","PDF + Voice","Downloadable reports & audio summaries"),
        ][0:4], [], []):
            pass
        feats = [
            ("🤖","Generative AI","Real Claude AI — not if-else rules"),
            ("👥","Collaboration","Team sessions with live chat & shared results"),
            ("📊","Auto ML","Auto-detects regression vs classification"),
            ("📄","PDF + Voice","Downloadable reports & audio summaries"),
        ]
        for col, (icon, title, desc) in zip(cols, feats):
            with col:
                st.markdown(f"""
                <div class="stat-card" style="text-align:left;padding:20px">
                  <div style="font-size:1.8rem;margin-bottom:8px">{icon}</div>
                  <div style="color:#58a6ff;font-weight:700;margin-bottom:4px">{title}</div>
                  <div style="color:#8b949e;font-size:0.82rem">{desc}</div>
                </div>""", unsafe_allow_html=True)
        st.info("👆 Upload a CSV from the sidebar to start your analysis.")
        return

    df = pd.read_csv(file)
    tab_data, tab_analysis, tab_visuals = st.tabs(["📂 Dataset", "🤖 Analysis", "📊 Visuals"])

    with tab_data:
        st.dataframe(df.head(20), use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",    df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", int(df.isnull().sum().sum()))
        c4.metric("Numeric", len(df.select_dtypes(include=np.number).columns))
        target = st.selectbox("🎯 Target Column", df.columns)
        st.session_state["_target_col"] = target

    with tab_analysis:
        target = st.session_state.get("_target_col", df.columns[0])
        X = pd.get_dummies(df.drop(columns=[target]))
        y = df[target]
        X = X.fillna(X.mean(numeric_only=True))
        try: y = pd.to_numeric(y)
        except: pass
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        problem = "classification" if y.nunique() < 10 else "regression"
        st.info(f"🔍 Detected: **{problem.title()}**")

        if st.button("🚀 Run ML Analysis", use_container_width=True):
            results, trained_models = [], {}
            with st.spinner("Training models…"):
                if problem == "regression":
                    models = {"Linear Regression": LinearRegression(),
                              "Random Forest":     RandomForestRegressor(n_estimators=100),
                              "Decision Tree":     DecisionTreeRegressor()}
                    metric = "R² Score"
                    for n, m in models.items():
                        m.fit(X_train, y_train)
                        sc = r2_score(y_test, m.predict(X_test))
                        results.append({"Model": n, metric: round(sc,4), "Score %": f"{sc*100:.2f}%"})
                        trained_models[n] = sc
                else:
                    models = {"Logistic Regression": LogisticRegression(max_iter=2000),
                              "Random Forest":       RandomForestClassifier(n_estimators=100),
                              "Decision Tree":       DecisionTreeClassifier()}
                    metric = "Accuracy"
                    for n, m in models.items():
                        m.fit(X_train, y_train)
                        sc = accuracy_score(y_test, m.predict(X_test))
                        results.append({"Model": n, metric: round(sc,4), "Score %": f"{sc*100:.2f}%"})
                        trained_models[n] = sc

            best_name  = max(trained_models, key=trained_models.get)
            best_score = trained_models[best_name]
            feasibility = ("Highly Feasible" if best_score > 0.85
                           else "Moderate"   if best_score > 0.65
                           else "Low")

            st.session_state.analysis_summary = {
                "best_model": best_name, "score": best_score,
                "problem": problem,      "feasibility": feasibility,
                "all_scores": trained_models,
                "n_rows": df.shape[0],   "n_cols": df.shape[1],
                "target": target,        "features": list(X.columns[:10])
            }

            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.success(f"🏆 **{best_name}**")
            c2.metric("Score", f"{best_score*100:.2f}%")
            c3.metric("Feasibility", feasibility)

            # ── Generative AI insight
            st.markdown("#### 🤖 AI Insight (Generated by Claude)")
            api_key = st.session_state.anthropic_key
            if api_key:
                with st.spinner("Generating AI insight…"):
                    prompt = (
                        f"Give expert analysis (4-6 sentences) of this ML result:\n"
                        f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols, target='{target}'\n"
                        f"Problem: {problem}, Best model: {best_name}, Score: {best_score*100:.1f}%\n"
                        f"All scores: {trained_models}, Feasibility: {feasibility}\n\n"
                        f"Cover: why this model likely won, practical meaning of the score, "
                        f"and 2-3 concrete next steps to improve."
                    )
                    system = build_system_prompt(st.session_state.analysis_summary, st.session_state.full_name)
                    ai_insight = call_claude([{"role":"user","content":prompt}], system, api_key)
                st.markdown(f'<div class="ai-insight"><div class="ai-insight-title">🤖 Claude\'s Analysis</div>{ai_insight}</div>',
                            unsafe_allow_html=True)
            else:
                ai_insight = "Add Anthropic API key in Settings for AI-generated insights."
                st.info(ai_insight)

            # ── Charts
            heatmap_path, dist_path = "heatmap.png", "dist.png"
            fig, ax = plt.subplots(figsize=(8,5))
            fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
            sns.heatmap(df.corr(numeric_only=True), ax=ax, cmap="Blues",
                        annot=True, fmt=".1f", linewidths=0.4, linecolor='#30363d',
                        annot_kws={"color":"#e6edf3","size":8})
            ax.tick_params(colors='#8b949e')
            plt.tight_layout(); plt.savefig(heatmap_path, facecolor='#161b22', bbox_inches='tight'); plt.close()

            fig2, ax2 = plt.subplots(figsize=(8,4))
            fig2.patch.set_facecolor('#161b22'); ax2.set_facecolor('#0d1117')
            sns.histplot(df[target], ax=ax2, color='#1f6feb', edgecolor='#58a6ff', bins=30)
            ax2.set_title(f"Distribution — {target}", color='#e6edf3')
            ax2.tick_params(colors='#8b949e'); ax2.spines[:].set_color('#30363d')
            plt.tight_layout(); plt.savefig(dist_path, facecolor='#161b22', bbox_inches='tight'); plt.close()

            # ── Voice
            try:
                tts = gTTS(f"Analysis done. Best model is {best_name}. "
                           f"Score {best_score*100:.1f} percent. Feasibility {feasibility}.")
                tts.save("voice.mp3"); st.audio("voice.mp3")
            except: pass

            # ── PDF
            try:
                pdf_path = "ML_Report.pdf"
                doc = SimpleDocTemplate(pdf_path)
                styles = getSampleStyleSheet()
                content = [
                    Paragraph("ML AI Advisor v3.0 — Analysis Report", styles['Title']),
                    Spacer(1,10),
                    Paragraph(f"Analyst: {st.session_state.full_name} (@{st.session_state.username})", styles['Normal']),
                    Paragraph(f"Problem: {problem.title()}", styles['Normal']),
                    Paragraph(f"Best Model: {best_name}", styles['Normal']),
                    Paragraph(f"Score: {best_score*100:.2f}%", styles['Normal']),
                    Paragraph(f"Feasibility: {feasibility}", styles['Normal']),
                    Spacer(1,12),
                    Paragraph("AI-Generated Insight (Claude):", styles['Heading2']),
                    Paragraph(ai_insight, styles['Normal']),
                    Spacer(1,12),
                    Paragraph("Correlation Heatmap", styles['Heading2']),
                    Image(heatmap_path, width=420, height=280),
                    Spacer(1,8),
                    Paragraph(f"Target Distribution — {target}", styles['Heading2']),
                    Image(dist_path, width=420, height=240),
                ]
                doc.build(content)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download PDF Report", f.read(),
                                       "ML_AI_Advisor_Report.pdf", "application/pdf",
                                       use_container_width=True)
            except Exception as e:
                st.warning(f"PDF error: {e}")

    with tab_visuals:
        fig, ax = plt.subplots(figsize=(10,6))
        fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
        sns.heatmap(df.corr(numeric_only=True), ax=ax, cmap="Blues",
                    annot=True, fmt=".1f", linewidths=0.4, linecolor='#30363d',
                    annot_kws={"color":"#e6edf3","size":8})
        ax.tick_params(colors='#8b949e')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            sel = st.selectbox("📈 Column distribution", num_cols)
            fig2, ax2 = plt.subplots(figsize=(8,4))
            fig2.patch.set_facecolor('#161b22'); ax2.set_facecolor('#0d1117')
            sns.histplot(df[sel], ax=ax2, color='#1f6feb', edgecolor='#58a6ff', bins=30)
            ax2.set_title(f"Distribution — {sel}", color='#e6edf3')
            ax2.tick_params(colors='#8b949e'); ax2.spines[:].set_color('#30363d')
            plt.tight_layout(); st.pyplot(fig2, use_container_width=True)


# ══════════════════════════════════════════════
#  ABOUT PAGE
# ══════════════════════════════════════════════
def page_about():
    st.markdown("""
## 🚀 ML AI Advisor — v3.0

### What's New in v3.0
| Feature | Details |
|---|---|
| 🤖 **Real Generative AI** | **Claude claude-sonnet-4-20250514** via Anthropic API — full language model, multi-turn memory, code generation |
| 👥 **Collaboration** | Create/join sessions · team chat · share results · team AI consult · comments |
| 📊 **AI-Generated Insights** | Claude writes expert recommendations from your actual ML results |
| 🔐 **Role-based Auth** | Student / Analyst / Researcher / Engineer roles |

### ⚙️ How to Enable Generative AI
1. Get a free API key at [console.anthropic.com](https://console.anthropic.com)
2. Paste it in **⚙️ Settings** in the sidebar
3. All AI features activate instantly — chatbot, analysis insights, team consult

### 🧠 Architecture
- **Generative AI**: Anthropic Claude claude-sonnet-4-20250514 REST API (real LLM, multi-turn)
- **ML Engine**: Scikit-learn (Linear, Logistic, Random Forest, Decision Tree)
- **Collaboration**: JSON-persisted sessions, live message feed, shared analysis
- **Reports**: ReportLab PDF · gTTS voice · Matplotlib/Seaborn charts

### 👨‍💻 Developer
**Nithin Mathew** · AIML Student · Future AI Engineer 🚀
© 2026 Nithin Mathew
""")


# ══════════════════════════════════════════════
#  MAIN SHELL
# ══════════════════════════════════════════════
def show_main_app():
    info  = st.session_state.user_info
    color = info.get("avatar_color", "#238636")

    # Sidebar
    st.sidebar.markdown("## 🚀 ML AI Advisor")
    st.sidebar.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
      <div style="background:{color};width:36px;height:36px;border-radius:50%;
           display:flex;align-items:center;justify-content:center;
           font-weight:700;font-size:0.9rem;color:#fff;flex-shrink:0">
        {st.session_state.full_name[:1].upper()}
      </div>
      <div>
        <div style="color:#e6edf3;font-weight:600;font-size:0.9rem">{st.session_state.full_name}</div>
        <div style="color:#8b949e;font-size:0.75rem">@{st.session_state.username} · {info.get('role','Analyst')}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    sid = st.session_state.collab_session_id
    if sid:
        sname = load_collab().get(sid, {}).get("name", sid)
        st.sidebar.markdown(f"""
        <div class="session-badge">👥 <strong>{sname}</strong><br>
        <code style="font-size:0.72rem">{sid}</code></div>""", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    page = st.sidebar.radio("📌 Navigate", [
        "🏠 Home", "🤖 AI Chatbot", "👥 Collaboration", "ℹ️ About"
    ])
    st.sidebar.markdown("---")

    with st.sidebar.expander("⚙️ Settings — API Key"):
        key_in = st.text_input("Anthropic API Key", value=st.session_state.anthropic_key,
                               type="password", placeholder="sk-ant-api03-…",
                               help="Get key at console.anthropic.com")
        if st.button("💾 Save Key", use_container_width=True):
            st.session_state.anthropic_key = key_in
            st.success("Saved for this session!")
        if st.session_state.anthropic_key:
            st.success("✅ API Key active")
        else:
            st.warning("⚠️ No key — AI limited")
        st.caption("[Get API key →](https://console.anthropic.com)")

    file = None
    if page == "🏠 Home":
        file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
    st.sidebar.caption("© 2026 Nithin Mathew · v3.0")

    # Top bar
    collab_tag = f" · 👥 {sid}" if sid else ""
    st.markdown(f"""
    <div class="top-bar">
      <div>
        <div class="top-bar-title">🚀 ML AI Advisor v3.0</div>
        <div class="top-bar-sub">Generative AI (Claude) · Collaboration · Auto ML{collab_tag}</div>
      </div>
      <div class="user-pill">
        <div style="background:{color};width:24px;height:24px;border-radius:50%;
             display:flex;align-items:center;justify-content:center;
             font-weight:700;font-size:0.7rem;color:#fff;flex-shrink:0">
          {st.session_state.full_name[:1].upper()}
        </div>
        {st.session_state.full_name}
      </div>
    </div>""", unsafe_allow_html=True)

    if   page == "🏠 Home":          page_home(file)
    elif page == "🤖 AI Chatbot":    page_chatbot()
    elif page == "👥 Collaboration": page_collaboration()
    elif page == "ℹ️ About":         page_about()


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 0">
      <div style="font-size:0.75rem;color:#484f58;letter-spacing:.1em;text-transform:uppercase">
        Generative AI · Collaboration · Auto ML
      </div>
    </div>""", unsafe_allow_html=True)
    if st.session_state.auth_page == "login":
        page_login()
    else:
        page_signup()
else:
    show_main_app()
