# ============================================================
# Student Exam Score Predictor — Colorful Vibrant UI
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide"
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* Hide default header */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 30%, #4facfe 70%, #00f2fe 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(240, 147, 251, 0.3);
    }
    .hero h1 {
        font-size: 3em;
        font-weight: 900;
        color: white;
        margin: 0;
        text-shadow: 2px 4px 10px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    .hero p {
        font-size: 1.1em;
        color: rgba(255,255,255,0.9);
        margin-top: 10px;
    }

    /* Section cards */
    .section-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Section headers */
    .section-title {
        font-size: 1.3em;
        font-weight: 700;
        margin-bottom: 20px;
        padding: 10px 16px;
        border-radius: 10px;
        display: inline-block;
    }
    .title-pink {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        color: white;
    }
    .title-blue {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white;
    }
    .title-green {
        background: linear-gradient(90deg, #43e97b, #38f9d7);
        color: white;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ff6b6b 100%);
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 15px 40px rgba(245, 87, 108, 0.4);
    }
    .result-score {
        font-size: 5em;
        font-weight: 900;
        color: white;
        text-shadow: 3px 5px 15px rgba(0,0,0,0.3);
        line-height: 1;
    }
    .result-label {
        font-size: 1.4em;
        color: rgba(255,255,255,0.9);
        margin-top: 8px;
        font-weight: 600;
    }
    .result-grade {
        font-size: 1.8em;
        font-weight: 800;
        color: white;
        margin-top: 10px;
    }

    /* Grade A box */
    .grade-a { background: linear-gradient(135deg, #43e97b, #38f9d7); }
    .grade-b { background: linear-gradient(135deg, #4facfe, #00f2fe); }
    .grade-c { background: linear-gradient(135deg, #f6d365, #fda085); }
    .grade-d { background: linear-gradient(135deg, #f093fb, #f5576c); }

    /* Stat cards */
    .stat-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 6px 0;
    }
    .stat-label {
        font-size: 0.8em;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        font-size: 1.5em;
        font-weight: 700;
        color: white;
        margin-top: 4px;
    }

    /* Progress bar custom */
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 50px;
        height: 16px;
        margin: 15px 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #43e97b, #38f9d7);
        transition: width 0.5s ease;
    }

    /* Tip cards */
    .tip-card {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #f093fb;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: rgba(255,255,255,0.85);
        font-size: 0.95em;
    }

    /* Override Streamlit widget label colors */
    .stSlider label, .stSelectbox label {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 500 !important;
    }

    /* Predict button */
    .stButton button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ff6b6b 100%) !important;
        color: white !important;
        font-size: 1.2em !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 16px 40px !important;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4) !important;
        transition: transform 0.2s !important;
        width: 100% !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 40px rgba(245, 87, 108, 0.6) !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("final_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("final_medians.pkl", "rb") as f:
        medians = pickle.load(f)
    return model, scaler, medians

model, scaler, medians = load_artifacts()

# ── Hero Banner ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎓 Student Score Predictor</h1>
    <p>Fill in the student profile below and instantly predict their exam score using AI</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Academic Info ─────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<span class="section-title title-pink">📚 Academic Information</span>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    hours_studied     = st.slider("Hours Studied / week", 1, 44, 20)
    previous_scores   = st.slider("Previous Scores", 50, 100, 70)
with col2:
    attendance        = st.slider("Attendance (%)", 60, 100, 80)
    tutoring_sessions = st.slider("Tutoring Sessions / month", 0, 8, 2)
with col3:
    motivation_level  = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    peer_influence    = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 2: Personal & Health ────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<span class="section-title title-blue">🏃 Personal & Health</span>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    sleep_hours       = st.slider("Sleep Hours / night", 4, 10, 7)
    physical_activity = st.slider("Physical Activity (days/week)", 0, 6, 3)
with col5:
    gender            = st.selectbox("Gender", ["Male", "Female"])
    learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
with col6:
    extracurricular   = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    internet_access   = st.selectbox("Internet Access", ["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Section 3: Background & School ──────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<span class="section-title title-green">🏫 Background & School</span>', unsafe_allow_html=True)

col7, col8, col9 = st.columns(3)
with col7:
    parental_involvement = st.selectbox("Parental Involvement",   ["Low", "Medium", "High"])
    access_to_resources  = st.selectbox("Access to Resources",    ["Low", "Medium", "High"])
with col8:
    family_income        = st.selectbox("Family Income",          ["Low", "Medium", "High"])
    teacher_quality      = st.selectbox("Teacher Quality",        ["Low", "Medium", "High"])
with col9:
    parental_education   = st.selectbox("Parental Education",     ["High School", "College", "Postgraduate"])
    distance_from_home   = st.selectbox("Distance from Home",     ["Near", "Moderate", "Far"])
    school_type          = st.selectbox("School Type",            ["Public", "Private"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("🎯 Predict Exam Score")

if predict:
    # Encoding maps
    ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
    peer_map    = {"Negative": 0, "Neutral": 1, "Positive": 2}
    edu_map     = {"High School": 0, "College": 1, "Postgraduate": 2}
    dist_map    = {"Near": 0, "Moderate": 1, "Far": 2}
    binary_map  = {"Yes": 1, "No": 0, "Male": 1, "Female": 0, "Public": 0, "Private": 1}

    raw = {
        "Hours_Studied":              hours_studied,
        "Attendance":                 attendance,
        "Parental_Involvement":       ordinal_map[parental_involvement],
        "Access_to_Resources":        ordinal_map[access_to_resources],
        "Extracurricular_Activities": binary_map[extracurricular],
        "Sleep_Hours":                sleep_hours,
        "Previous_Scores":            previous_scores,
        "Motivation_Level":           ordinal_map[motivation_level],
        "Internet_Access":            binary_map[internet_access],
        "Tutoring_Sessions":          tutoring_sessions,
        "Family_Income":              ordinal_map[family_income],
        "Teacher_Quality":            ordinal_map[teacher_quality],
        "School_Type":                binary_map[school_type],
        "Peer_Influence":             peer_map[peer_influence],
        "Physical_Activity":          physical_activity,
        "Learning_Disabilities":      binary_map[learning_disabilities],
        "Parental_Education_Level":   edu_map[parental_education],
        "Distance_from_Home":         dist_map[distance_from_home],
        "Gender":                     binary_map[gender],
    }

    df = pd.DataFrame([raw])
    df["Study_Efficiency"]      = df["Hours_Studied"] / (df["Attendance"] + 1)
    df["Support_Score"]         = (df["Parental_Involvement"] + df["Access_to_Resources"] +
                                    df["Teacher_Quality"] + df["Internet_Access"])
    df["Wellbeing_Score"]       = df["Sleep_Hours"] + df["Physical_Activity"]
    df["Academic_History"]      = df["Previous_Scores"] * 0.7 + df["Tutoring_Sessions"] * 0.3
    df["Motivation_Resources"]  = df["Motivation_Level"] * df["Access_to_Resources"]
    df["Attendance_Study"]      = df["Attendance"] * df["Hours_Studied"]

    df = df.fillna(medians)
    df_scaled = scaler.transform(df)
    score = float(np.clip(model.predict(df_scaled)[0], 55, 100))

    # Grade
    if score >= 80:
        grade, emoji, grade_class = "Excellent", "🏆", "grade-a"
    elif score >= 70:
        grade, emoji, grade_class = "Good", "✅", "grade-b"
    elif score >= 60:
        grade, emoji, grade_class = "Average", "📈", "grade-c"
    else:
        grade, emoji, grade_class = "Needs Improvement", "⚠️", "grade-d"

    # ── Results Layout ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    res_col, stats_col = st.columns([1, 1])

    with res_col:
        st.markdown(f"""
        <div class="result-box {grade_class}">
            <div class="result-score">{score:.1f}</div>
            <div class="result-label">out of 100</div>
            <div class="result-grade">{emoji} {grade}</div>
        </div>
        <div class="progress-container">
            <div class="progress-fill" style="width:{score}%"></div>
        </div>
        """, unsafe_allow_html=True)

    with stats_col:
        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Study Hours</div><div class="stat-value">{hours_studied}h</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-label">Attendance</div><div class="stat-value">{attendance}%</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-label">Sleep</div><div class="stat-value">{sleep_hours}h</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Prev. Score</div><div class="stat-value">{previous_scores}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-label">Tutoring</div><div class="stat-value">{tutoring_sessions}x</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stat-card"><div class="stat-label">Activity</div><div class="stat-value">{physical_activity}d</div></div>', unsafe_allow_html=True)

    # ── Tips ─────────────────────────────────────────────────
    tips = []
    if hours_studied < 20:
        tips.append("📚 Increase study hours — aim for at least 20 hours per week.")
    if attendance < 80:
        tips.append("🏫 Improve attendance — try to attend at least 80% of classes.")
    if sleep_hours < 7:
        tips.append("😴 Get more sleep — 7–9 hours improves memory and focus.")
    if tutoring_sessions == 0:
        tips.append("👨‍🏫 Consider tutoring sessions for extra support.")
    if motivation_level == "Low":
        tips.append("🎯 Work on motivation — set small daily goals to build momentum.")
    if physical_activity < 3:
        tips.append("🏃 Exercise more — physical activity boosts concentration.")

    if tips:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<span class="section-title title-pink">💡 Tips to Improve</span>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tip-card" style="border-left-color: #43e97b;">
            🌟 Outstanding profile! Keep up the excellent habits.
        </div>
        """, unsafe_allow_html=True)