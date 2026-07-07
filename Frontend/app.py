

import warnings
warnings.filterwarnings("ignore")

import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "Backend", ".env"))

import streamlit as st
from pipeline import pipeline, parse_pdf_bytes



st.set_page_config(
    page_title="Resume–JD Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume–JD Matching Tool")
st.markdown("Upload or paste your **Resume** and **Job Description**, then click **Match**.")
st.divider()



col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Resume")
    uploaded_resume_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])
    resume_text_input = st.text_area(
        "Or paste resume text here",
        height=250,
        placeholder="John Doe\nPython Developer\n\nSkills: Python, FastAPI..."
    )

    if uploaded_resume_pdf is not None:
        resume_input = parse_pdf_bytes(uploaded_resume_pdf.read())
        st.success(f"✅ Resume PDF loaded: {uploaded_resume_pdf.name}")
        with st.expander("See extracted text"):
            st.text(resume_input[:1000] + "..." if len(resume_input) > 1000 else resume_input)
    else:
        resume_input = resume_text_input


with col2:
    st.subheader("📝 Job Description")
    uploaded_jd_pdf = st.file_uploader("Upload Job Description PDF", type=["pdf"])
    jd_text_input = st.text_area(
        "Or paste job description here",
        height=250,
        placeholder="We are hiring a Senior Python Engineer...\n\nRequirements:\n- Python\n- FastAPI..."
    )

    if uploaded_jd_pdf is not None:
        jd_input = parse_pdf_bytes(uploaded_jd_pdf.read())
        st.success(f"✅ JD PDF loaded: {uploaded_jd_pdf.name}")
        with st.expander("See extracted text"):
            st.text(jd_input[:1000] + "..." if len(jd_input) > 1000 else jd_input)
    else:
        jd_input = jd_text_input

st.divider()



if st.button("🔍 Match Resume to JD", use_container_width=True, type="primary"):

    if not resume_input or not resume_input.strip():
        st.warning("Please upload a PDF or paste resume text.")
    elif not jd_input.strip():
        st.warning("Please paste the Job Description.")
    else:
        with st.spinner("Analysing with Mistral AI..."):
            result = pipeline.invoke({"resume": resume_input, "jd": jd_input})

        st.divider()
        st.subheader("📊 Results")

        m1, m2 = st.columns(2)
        m1.metric("Match Score",         f"{result.get('score', 0)} / 100")
        m2.metric("Semantic Similarity",  f"{result.get('similarity', 0)}")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ Matched Skills")
            for skill in result.get("matched_skills", []):
                st.success(skill)

        with c2:
            st.markdown("### ❌ Missing Skills")
            for skill in result.get("missing_skills", []):
                st.error(skill)

        st.divider()

        st.markdown("### 💡 Suggestions")
        for s in result.get("suggestions", []):
            st.info(s)

        st.divider()

        st.markdown("### 🎯 Overall Recommendation")
        st.success(result.get("overall_recommendation", ""))
