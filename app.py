"""
Nepal Hospital Finder - Streamlit App
AI-Powered Hospital Recommendation System
ST5000CEM - Introduction to Artificial Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import time

sys.path.append('src')

from disease_classifier import DiseaseClassifier, DISEASE_SPECIALTY_MAP
from hospital_recommender import HospitalRecommender

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* ── Header ── */
    .app-title {
        font-size: 36px;
        font-weight: 700;
        color: #1a7a4a;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .app-sub {
        font-size: 15px;
        color: #6b7280;
        text-align: center;
        margin-bottom: 24px;
    }

    /* ── Divider ── */
    .green-line {
        height: 3px;
        background: linear-gradient(to right, #1a7a4a, #4ade80, #1a7a4a);
        border-radius: 2px;
        margin: 16px 0 28px 0;
    }

    /* ── Info card ── */
    .info-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    .info-card h4 {
        color: #166534;
        margin: 0 0 8px 0;
        font-size: 15px;
    }
    .info-card p {
        color: #374151;
        margin: 0;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ── Feature grid ── */
    .feat-box {
        background: #ffffff;
        border: 1px solid #d1fae5;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
        height: 100%;
    }
    .feat-box .icon { font-size: 24px; margin-bottom: 6px; }
    .feat-box .label {
        font-size: 13px;
        font-weight: 600;
        color: #166534;
        margin-bottom: 4px;
    }
    .feat-box .desc { font-size: 12px; color: #6b7280; }

    /* ── Section heading ── */
    .section-heading {
        font-size: 16px;
        font-weight: 600;
        color: #166534;
        border-left: 4px solid #22c55e;
        padding-left: 10px;
        margin: 24px 0 12px 0;
    }

    /* ── Hospital card ── */
    .hosp-card {
        background: #ffffff;
        border: 1px solid #d1fae5;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #22c55e;
    }
    .hosp-name {
        font-size: 16px;
        font-weight: 700;
        color: #166534;
        margin-bottom: 6px;
    }
    .hosp-detail {
        font-size: 13px;
        color: #374151;
        margin: 3px 0;
    }
    .badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        margin-right: 4px;
        margin-bottom: 6px;
    }
    .badge-green  { background:#dcfce7; color:#166534; }
    .badge-blue   { background:#dbeafe; color:#1e40af; }
    .badge-yellow { background:#fef9c3; color:#854d0e; }
    .badge-red    { background:#fee2e2; color:#991b1b; }
    .badge-gray   { background:#f3f4f6; color:#374151; }

    /* ── Result summary card ── */
    .result-top {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .result-disease {
        font-size: 22px;
        font-weight: 700;
        color: #166534;
    }
    .result-meta {
        font-size: 13px;
        color: #6b7280;
        margin-top: 6px;
    }

    /* ── Step card ── */
    .step-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .step-title { font-weight: 600; color: #166534; font-size: 14px; }
    .step-body  { font-size: 13px; color: #374151; margin-top: 4px; }
    .step-time  { font-size: 11px; color: #9ca3af; margin-top: 4px; }

    /* ── Button override ── */
    .stButton > button {
        background-color: #16a34a !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stButton > button:hover {
        background-color: #15803d !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background-color: #22c55e !important;
    }

    /* ── Metric ── */
    [data-testid="metric-container"] {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 12px;
    }

    /* hide sidebar toggle on welcome */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────
for key, val in [('page','welcome'),('user_profile',{}),
                 ('prediction',None),('hospitals',None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─── LOADERS ─────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    try:
        return DiseaseClassifier.load()
    except FileNotFoundError:
        clf = DiseaseClassifier(); clf.train(); return clf

@st.cache_resource
def load_recommender():
    return HospitalRecommender().load()

@st.cache_data
def load_precautions():
    df = pd.read_csv('data/raw/symptom_precaution.csv')
    df['Disease'] = df['Disease'].str.strip()
    return df.set_index('Disease')

@st.cache_data
def load_descriptions():
    df = pd.read_csv('data/raw/symptom_Description.csv')
    df['Disease'] = df['Disease'].str.strip()
    return df.set_index('Disease')

# ─── DATA ────────────────────────────────────────────────────
DISTRICTS = [
    "Any in Nepal","Kathmandu","Lalitpur","Bhaktapur",
    "Pokhara","Biratnagar","Birgunj","Dharan","Butwal",
    "Nepalgunj","Hetauda","Chitwan","Itahari","Dhangadhi",
    "Janakpur","Gorkha",
]

SYMPTOM_GROUPS = {
    "🤒 Fever & General": [
        'high_fever','mild_fever','chills','shivering','sweating',
        'fatigue','lethargy','malaise','dehydration','weight_loss','weight_gain',
    ],
    "🫁 Respiratory": [
        'cough','breathlessness','phlegm','blood_in_sputum','rusty_sputum',
        'mucoid_sputum','congestion','runny_nose','continuous_sneezing',
        'throat_irritation','sinus_pressure',
    ],
    "🫀 Heart & Chest": ['chest_pain','fast_heart_rate','palpitations'],
    "🤢 Stomach & Digestion": [
        'stomach_pain','abdominal_pain','nausea','vomiting','diarrhoea',
        'constipation','indigestion','acidity','loss_of_appetite','belly_pain',
        'passage_of_gases','stomach_bleeding','distention_of_abdomen',
    ],
    "🧠 Head & Neurological": [
        'headache','dizziness','spinning_movements','loss_of_balance',
        'unsteadiness','stiff_neck','slurred_speech','weakness_of_one_body_side',
        'altered_sensorium','lack_of_concentration','visual_disturbances',
        'blurred_and_distorted_vision','loss_of_smell',
    ],
    "🦴 Joints & Muscles": [
        'joint_pain','back_pain','neck_pain','knee_pain','hip_joint_pain',
        'muscle_pain','muscle_weakness','muscle_wasting','movement_stiffness',
        'swelling_joints','painful_walking','cramps',
    ],
    "🩺 Skin": [
        'itching','skin_rash','nodal_skin_eruptions','yellowish_skin',
        'red_spots_over_body','blackheads','pus_filled_pimples','skin_peeling',
        'blister','dischromic_patches','scurring','silver_like_dusting',
    ],
    "💛 Liver & Jaundice": [
        'yellowing_of_eyes','yellow_urine','dark_urine',
        'acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes',
    ],
    "🩸 Blood & Sugar": [
        'excessive_hunger','increased_appetite','polyuria','irregular_sugar_level',
        'bruising','swollen_blood_vessels','prominent_veins_on_calf',
    ],
    "🧬 Other": [
        'anxiety','depression','irritability','mood_swings','restlessness',
        'cold_hands_and_feets','puffy_face_and_eyes','enlarged_thyroid',
        'swollen_legs','swollen_extremeties','obesity',
        'burning_micturition','continuous_feel_of_urine','bladder_discomfort',
    ],
}

# ─────────────────────────────────────────────────────────────
def header(title, sub=None):
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="app-sub">{sub}</div>', unsafe_allow_html=True)
    st.markdown('<div class="green-line"></div>', unsafe_allow_html=True)

def section(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — WELCOME
# ═══════════════════════════════════════════════════════════════
def welcome_page():
    header("🏥Disease Predictor",
           "Predict the likely disease based on your symptoms")

    st.markdown("""
    <div class="info-card">
        <h4>How it works</h4>
        <p>
        Tell us your symptoms and preferences, and our AI will predict the most likely condition you have.
        Our AI analyses your symptoms, predicts the likely condition,
        and identifies the required medical specialty.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    feats = [
        ("🔬", "Symptom Analysis", "Select from 131 clinical symptoms"),
        ("🧠", "AI Prediction", "RandomForest disease classifier"),
        ("🏥", "Hospital Match", "TF-IDF similarity ranking"),
        ("💰", "Budget Filter", "Finds affordable options for you"),
    ]
    for col, (icon, label, desc) in zip([c1,c2,c3,c4], feats):
        col.markdown(f"""
        <div class="feat-box">
            <div class="icon">{icon}</div>
            <div class="label">{label}</div>
            <div class="desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.page = 'questionnaire'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("ST5000CEM – Introduction to Artificial Intelligence | Softwarica College of IT & E-Commerce | Coventry University")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — QUESTIONNAIRE
# ═══════════════════════════════════════════════════════════════
def questionnaire_page():
    header("📝 Your Symptoms & Preferences")

    # ── Symptoms (outside form) ───────────────────────────────
    section("1 — Select Your Symptoms")
    st.caption("Expand each group and tick everything you are experiencing.")

    selected_symptoms = []
    for group_name, symptoms in SYMPTOM_GROUPS.items():
        with st.expander(group_name, expanded=False):
            cols = st.columns(3)
            for i, sym in enumerate(symptoms):
                if cols[i % 3].checkbox(sym.replace("_"," ").title(), key=f"sym_{sym}"):
                    selected_symptoms.append(sym)

    if selected_symptoms:
        names = ", ".join(s.replace("_"," ").title() for s in selected_symptoms[:6])
        extra = f" +{len(selected_symptoms)-6} more" if len(selected_symptoms) > 6 else ""
        st.success(f"✅  **{len(selected_symptoms)} selected:** {names}{extra}")
    else:
        st.info("☝️  Expand the groups above and select your symptoms.")

    st.divider()

    # ── Preferences (inside form) ─────────────────────────────
    section("2 — Budget & Location")
    with st.form("patient_form"):

        c1, c2 = st.columns(2)
        with c1:
            budget_option = st.select_slider(
                "Maximum OPD fee (NPR)",
                options=["Less than 300","300 - 600","600 - 1,000",
                         "1,000 - 1,500","1,500 - 2,500","Above 2,500"],
            )
            preferred_district = st.selectbox("Preferred district", DISTRICTS)

        with c2:
            is_emergency = st.radio(
                "Visit type",
                ["Routine visit", "Emergency / Urgent"],
            )
            ownership_pref = st.multiselect(
                "Hospital type (optional)",
                ["Government","Private","Community/NGO"],
            )

        st.divider()
        section("3 — Additional")

        c1, c2, c3 = st.columns(3)
        need_ambulance = c1.checkbox("Ambulance needed")
        need_parking   = c2.checkbox("Parking required")
        top_n          = c3.slider("Results to show", 3, 10, 5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  Predict Disease", use_container_width=True)

        if submitted:
            if not selected_symptoms:
                st.error("❌  Please select at least one symptom first.")
                return

            budget_map = {
                "Less than 300":300,"300 - 600":600,"600 - 1,000":1000,
                "1,000 - 1,500":1500,"1,500 - 2,500":2500,"Above 2,500":5000,
            }
            st.session_state.user_profile = {
                "symptoms":       selected_symptoms,
                "budget_npr":     budget_map[budget_option],
                "district":       "" if preferred_district == "Any in Nepal" else preferred_district,
                "emergency":      "Emergency" in is_emergency,
                "ownership_pref": ownership_pref,
                "need_ambulance": need_ambulance,
                "need_parking":   need_parking,
                "top_n":          top_n,
            }
            st.session_state.page = "processing"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — PROCESSING
# ═══════════════════════════════════════════════════════════════
def processing_page():
    header("🤖 Analysing Your Profile...")

    bar  = st.progress(0)
    slot = st.empty()

    steps = [
        ("Analysing symptoms...", 20),
        ("Running disease classifier...", 42),
        ("Identifying required specialty...", 58),
        ("Matching hospitals with TF-IDF...", 72),
        ("Applying budget & location filters...", 86),
        ("Ranking by match score...", 95),
        ("Done!", 100),
    ]
    for msg, pct in steps:
        slot.markdown(f"**{msg}**")
        bar.progress(pct)
        time.sleep(0.35)

    try:
        clf  = load_classifier()
        rec  = load_recommender()
        prof = st.session_state.user_profile

        prediction = clf.predict(prof['symptoms'])
        hospitals  = rec.recommend(
            specialty  = prediction['specialty'],
            budget_npr = prof['budget_npr'],
            district   = prof['district'],
            emergency  = prof['emergency'],
            top_n      = prof['top_n'],
        )

        if prof['ownership_pref']:
            f = hospitals[hospitals['ownership_type'].isin(prof['ownership_pref'])]
            hospitals = f if not f.empty else hospitals

        if prof['need_ambulance']:
            f = hospitals[hospitals['ambulance_service'] == True]
            hospitals = f if not f.empty else hospitals

        st.session_state.prediction = prediction
        st.session_state.hospitals  = hospitals
        st.session_state.page       = 'results'
        st.rerun()

    except Exception as e:
        st.error(f"❌  {str(e)}")
        st.error("Make sure setup_and_train.py has been run first.")
        if st.button("← Go Back"):
            st.session_state.page = 'questionnaire'
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ═══════════════════════════════════════════════════════════════
def results_page():
    if st.session_state.prediction is None:
        st.error("No results. Please go back.")
        if st.button("← Back"):
            st.session_state.page = 'questionnaire'; st.rerun()
        return

    pred     = st.session_state.prediction
    hospitals= st.session_state.hospitals
    profile  = st.session_state.user_profile
    prec_df  = load_precautions()
    desc_df  = load_descriptions()

    disease   = pred['disease']
    specialty = pred['specialty']
    conf      = pred['confidence']

    header("🎯 Your Results")

    # ── Top result card ───────────────────────────────────────
    st.markdown(f"""
    <div class="result-top">
        <div class="result-disease">🔬 {disease}</div>
        <div class="result-meta">
            Specialty needed: <strong>{specialty}</strong> &nbsp;·&nbsp;
            Confidence: <strong>{conf*100:.1f}%</strong> &nbsp;·&nbsp;
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Description ───────────────────────────────────────────
    if disease in desc_df.index:
        desc = str(desc_df.loc[disease, 'Description'])
        st.caption(desc[:200] + ("..." if len(desc) > 200 else ""))

    # ── Overview columns ──────────────────────────────────────
    section("Overview")
    c1, c2 = st.columns([1, 1])

    with c1:
        syms = ", ".join(s.replace("_"," ").title() for s in profile['symptoms'][:5])
        if len(profile['symptoms']) > 5:
            syms += f" +{len(profile['symptoms'])-5} more"
        st.markdown(f"**Symptoms entered:** {syms}")
        st.markdown(f"**Predicted condition:** {disease}")
        st.markdown(f"**Required specialist:** {specialty}")

    with c2:
        top5   = pred['top5']
        labels = [d for d,_ in top5]
        values = [round(p*100,2) for _,p in top5]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.55,
            marker=dict(colors=['#16a34a','#4ade80','#86efac','#bbf7d0','#dcfce7']),
            textinfo='percent', textfont_size=11,
        ))
        fig.update_layout(
            height=220, margin=dict(l=0,r=0,t=0,b=0),
            showlegend=True,
            legend=dict(font=dict(size=10)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Precautions ───────────────────────────────────────────
    if disease in prec_df.index:
        section("Immediate Precautions")
        precs = [str(prec_df.loc[disease, f'Precaution_{i}'])
                 for i in range(1,5)
                 if pd.notna(prec_df.loc[disease, f'Precaution_{i}'])]
        cols = st.columns(len(precs))
        for col, p in zip(cols, precs):
            col.success(f"✅ {p.capitalize()}")

    # ── Hospitals ─────────────────────────────────────────────
    section(f"Recommended Hospitals ({len(hospitals)} found)")

    if hospitals is not None and not hospitals.empty:

        fc1, fc2, fc3 = st.columns(3)
        show_count      = fc1.selectbox("Show", [5, 10, len(hospitals)], index=0, label_visibility="collapsed")
        only_emergency  = fc2.checkbox("Emergency only")
        only_ambulance  = fc3.checkbox("Ambulance only")

        df = hospitals.copy()
        if only_emergency: df = df[df['emergency_available']==True]
        if only_ambulance: df = df[df['ambulance_service']==True]
        if df.empty: df = hospitals

        for i, (_, row) in enumerate(df.head(show_count).iterrows(), 1):
            tier_badge = (
                '<span class="badge badge-green">⭐ Premium</span>' if row['tier']=='Premium'
                else '<span class="badge badge-blue">Standard</span>' if row['tier']=='Standard'
                else '<span class="badge badge-yellow">Budget</span>'
            )
            emerg_badge  = '<span class="badge badge-red">🚨 Emergency</span>'   if row['emergency_available'] else ''
            ambul_badge  = '<span class="badge badge-blue">🚑 Ambulance</span>'  if row['ambulance_service']   else ''
            own_badge    = (
                '<span class="badge badge-green">🏛 Government</span>'  if row['ownership_type']=='Government'
                else '<span class="badge badge-gray">🤝 NGO</span>'     if row['ownership_type']=='Community/NGO'
                else '<span class="badge badge-gray">🏢 Private</span>'
            )
            budget_ok = profile['budget_npr'] >= row['opd_fee_npr']
            fee_color = "#166534" if budget_ok else "#991b1b"
            fee_icon  = "✅" if budget_ok else "⚠️"

            specs = str(row['specialties'])[:70] + ("..." if len(str(row['specialties']))>70 else "")

            st.markdown(f"""
            <div class="hosp-card">
                <div class="hosp-name">#{i}. {row['hospital_name']}</div>
                <div style="margin-bottom:8px">
                    {tier_badge}{emerg_badge}{ambul_badge}{own_badge}
                </div>
                <div class="hosp-detail">📍 {row['district']}</div>
                <div class="hosp-detail">🔬 {specs}</div>
                <div class="hosp-detail" style="color:{fee_color}">
                    {fee_icon} OPD Fee: NPR {int(row['opd_fee_npr']):,}
                </div>
                <div class="hosp-detail">⭐ Rating: {row['rating']}/5.0
                    &nbsp;·&nbsp; 🛏 {int(row['bed_capacity'])} beds
                    &nbsp;·&nbsp; Score: {row['score']:.1f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Cost summary ──────────────────────────────────────
        section("Cost Summary")
        m1, m2, m3, m4 = st.columns(4)
        top_fee = int(df.iloc[0]['opd_fee_npr'])
        m1.metric("Your Budget",       f"NPR {profile['budget_npr']:,}")
        m2.metric("Top Hospital Fee",  f"NPR {top_fee:,}")
        m3.metric("Cheapest Option",   f"NPR {int(df['opd_fee_npr'].min()):,}")
        affordable = int((df['opd_fee_npr'] <= profile['budget_npr']).sum())
        m4.metric("Affordable",        f"{affordable} / {len(df)}")

    else:
        st.warning("No hospitals matched your criteria. Try relaxing your budget or location.")

    # ── Next steps ────────────────────────────────────────────
    section("Next Steps")

    steps = []
    if profile.get('emergency'):
        steps.append(("🚨 Emergency — Act Now",
                       "Call 102 (ambulance) or go directly to the nearest emergency department.",
                       "RIGHT NOW"))
    steps += [
        ("1️⃣ Visit the Top Hospital",
         f"Head to {hospitals.iloc[0]['hospital_name']} in {hospitals.iloc[0]['district']} — ask for the {specialty} department.",
         "As soon as possible"),
        ("2️⃣ Bring Your Documents",
         "Carry previous prescriptions, test reports, citizenship card, and health insurance (if any).",
         "Before visiting"),
        ("3️⃣ Follow Precautions",
         "Follow the precautions listed above while you wait for your appointment.",
         "Immediately"),
    ]

    for title, body, timeline in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-title">{title}</div>
            <div class="step-body">{body}</div>
            <div class="step-time">⏱ {timeline}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Alternatives ──────────────────────────────────────────
    if len(pred['top5']) > 1:
        section("Other Possible Conditions")
        for i, (alt, prob) in enumerate(pred['top5'][1:4], 2):
            alt_spec = DISEASE_SPECIALTY_MAP.get(alt.strip(), 'General Medicine')
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"**{i}. {alt}** — *{alt_spec}*")
            c2.markdown(f"**{prob*100:.1f}%**")

    # ── Nav ───────────────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Back to Form", use_container_width=True):
            st.session_state.page = 'questionnaire'; st.rerun()
    with c3:
        if st.button("🔄 Start Over", use_container_width=True):
            for k,v in [('page','welcome'),('user_profile',{}),
                        ('prediction',None),('hospitals',None)]:
                st.session_state[k] = v
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    page = st.session_state.page
    if   page == 'welcome':       welcome_page()
    elif page == 'questionnaire': questionnaire_page()
    elif page == 'processing':    processing_page()
    elif page == 'results':       results_page()

if __name__ == "__main__":
    main()