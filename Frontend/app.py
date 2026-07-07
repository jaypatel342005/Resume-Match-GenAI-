import warnings
import sys
import os
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Backend"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "Backend", ".env"))

from pipeline import pipeline, parse_pdf_bytes

st.set_page_config(
    page_title="Resume Matcher Pro",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #2b5876, #4e4376);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Resume Matcher Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Evaluate candidates against job descriptions with GenAI</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Candidate Resume")
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    resume_text = st.text_area(
        "Or paste raw resume text here",
        height=300,
        placeholder="Name: John Doe\n\nExperience: 5 years Python Dev..."
    )

    if uploaded_resume is not None:
        resume_val = parse_pdf_bytes(uploaded_resume.read())
        st.success(f"Loaded: {uploaded_resume.name}")
        with st.expander("View parsed text preview"):
            st.text(resume_val[:1000] + "..." if len(resume_val) > 1000 else resume_val)
    else:
        resume_val = resume_text

with col2:
    st.subheader("Job Description")
    uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    jd_text = st.text_area(
        "Or paste job requirements here",
        height=300,
        placeholder="Requirements:\n- Python, FastAPI\n- PostgreSQL..."
    )

    if uploaded_jd is not None:
        jd_val = parse_pdf_bytes(uploaded_jd.read())
        st.success(f"Loaded: {uploaded_jd.name}")
        with st.expander("View parsed text preview"):
            st.text(jd_val[:1000] + "..." if len(jd_val) > 1000 else jd_val)
    else:
        jd_val = jd_text

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Run Profile Analysis", use_container_width=True, type="primary"):
    if not resume_val or not resume_val.strip():
        st.warning("Please upload a resume or paste text first.")
    elif not jd_val or not jd_val.strip():
        st.warning("Please upload a job description or paste requirements.")
    else:
        with st.spinner("Processing documents with LLM..."):
            res = pipeline.invoke({"resume": resume_val, "jd": jd_val})

        st.subheader("Evaluation Results")
        
        m1, m2 = st.columns(2)
        m1.metric("Overall Match Score", f"{res.get('score', 0)} / 100")
        m2.metric("Profile Similarity", f"{res.get('similarity', 0)}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Match Highlights")
            for skill in res.get("matched_skills", []):
                st.success(skill)

        with c2:
            st.markdown("### Missing Requirements")
            for skill in res.get("missing_skills", []):
                st.error(skill)

        st.markdown("---")
        
        st.markdown("### Gap Analysis & Suggestions")
        for sug in res.get("suggestions", []):
            st.info(sug)

        st.markdown("---")
        
        st.markdown("### Fit Summary & Verdict")
        st.warning(res.get("overall_recommendation", ""))
