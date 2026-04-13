# ============================================================
# Streamlit Web App — Student Exam Score Predictor (Fixed)
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
    layout="centered"
)

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

# ── Header ───────────────────────────────────────────────────
st.title("🎓 Student Exam Score Predictor")
st.markdown("Fill in the student details below and click **Predict** to estimate their exam score.")
st.divider()

# ── Input Form ───────────────────────────────────────────────
st.subheader("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    hours_studied       = st.slider("Hours Studied per week", 1, 44, 20)
    attendance          = st.slider("Attendance (%)", 60, 100, 80)
    sleep_hours         = st.slider("Sleep Hours per night", 4, 10, 7)
    previous_scores     = st.slider("Previous Scores", 50, 100, 70)
    tutoring_sessions   = st.slider("Tutoring Sessions per month", 0, 8, 2)
    physical_activity   = st.slider("Physical Activity (days/week)", 0, 6, 3)

with col2:
    parental_involvement    = st.selectbox("Parental Involvement",    ["Low", "Medium", "High"])
    access_to_resources     = st.selectbox("Access to Resources",     ["Low", "Medium", "High"])
    motivation_level        = st.selectbox("Motivation Level",        ["Low", "Medium", "High"])
    family_income           = st.selectbox("Family Income",           ["Low", "Medium", "High"])
    teacher_quality         = st.selectbox("Teacher Quality",         ["Low", "Medium", "High"])
    peer_influence          = st.selectbox("Peer Influence",          ["Negative", "Neutral", "Positive"])

st.divider()
st.subheader("🏫 School & Background")

col3, col4 = st.columns(2)

with col3:
    parental_education  = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
    distance_from_home  = st.selectbox("Distance from Home",       ["Near", "Moderate", "Far"])
    school_type         = st.selectbox("School Type",              ["Public", "Private"])

with col4:
    gender                      = st.selectbox("Gender",                     ["Male", "Female"])
    extracurricular_activities  = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    internet_access             = st.selectbox("Internet Access",            ["Yes", "No"])
    learning_disabilities       = st.selectbox("Learning Disabilities",      ["No", "Yes"])

st.divider()

# ── Predict button ───────────────────────────────────────────
if st.button("🎯 Predict Exam Score", use_container_width=True, type="primary"):

    # Encoding maps
    ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
    peer_map    = {"Negative": 0, "Neutral": 1, "Positive": 2}
    edu_map     = {"High School": 0, "College": 1, "Postgraduate": 2}
    dist_map    = {"Near": 0, "Moderate": 1, "Far": 2}
    binary_map  = {"Yes": 1, "No": 0, "Male": 1, "Female": 0, "Public": 0, "Private": 1}

    # Build raw feature dict
    raw = {
        "Hours_Studied":              hours_studied,
        "Attendance":                 attendance,
        "Parental_Involvement":       ordinal_map[parental_involvement],
        "Access_to_Resources":        ordinal_map[access_to_resources],
        "Extracurricular_Activities": binary_map[extracurricular_activities],
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

    # Feature engineering
    df = pd.DataFrame([raw])
    df["Study_Efficiency"]      = df["Hours_Studied"] / (df["Attendance"] + 1)
    df["Support_Score"]         = (df["Parental_Involvement"] + df["Access_to_Resources"] +
                                    df["Teacher_Quality"] + df["Internet_Access"])
    df["Wellbeing_Score"]       = df["Sleep_Hours"] + df["Physical_Activity"]
    df["Academic_History"]      = df["Previous_Scores"] * 0.7 + df["Tutoring_Sessions"] * 0.3
    df["Motivation_Resources"]  = df["Motivation_Level"] * df["Access_to_Resources"]
    df["Attendance_Study"]      = df["Attendance"] * df["Hours_Studied"]

    # Fill NaNs + scale
    df = df.fillna(medians)
    df_scaled = scaler.transform(df)

    # Predict
    score = model.predict(df_scaled)[0]
    score = float(np.clip(score, 55, 100))

    # ── Result display ───────────────────────────────────────
    st.subheader("📊 Prediction Result")

    if score >= 80:
        grade, emoji = "A — Excellent", "🏆"
    elif score >= 70:
        grade, emoji = "B — Good", "✅"
    elif score >= 60:
        grade, emoji = "C — Average", "📈"
    else:
        grade, emoji = "D — Needs Improvement", "⚠️"

    st.metric(label="Predicted Exam Score", value=f"{score:.1f} / 100")
    st.markdown(f"**Grade: {emoji} {grade}**")
    st.progress(int(score))

    # Key factors — all values as strings to avoid PyArrow error
    st.divider()
    st.subheader("🔍 Key Factors Summary")
    factors = {
        "Factor": ["Study Hours", "Attendance", "Previous Scores",
                   "Support Score", "Wellbeing Score", "Motivation"],
        "Value":  [
            str(hours_studied),
            f"{attendance}%",
            str(previous_scores),
            str(ordinal_map[parental_involvement] + ordinal_map[access_to_resources] +
                ordinal_map[teacher_quality] + binary_map[internet_access]),
            str(sleep_hours + physical_activity),
            motivation_level
        ]
    }
    st.dataframe(pd.DataFrame(factors), use_container_width=True, hide_index=True)

    # Tips
    st.divider()
    st.subheader("💡 Tips to Improve")
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
        for tip in tips:
            st.markdown(tip)
    else:
        st.success("🌟 Great profile! Keep up the excellent habits.")