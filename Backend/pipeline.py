import os
import tempfile
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

load_dotenv()

llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
embeddings_model = MistralAIEmbeddings(model="mistral-embed")

skill_prompt = ChatPromptTemplate.from_template("""
Extract skills from the document below and categorize them into JSON.

Categories:
- technical_skills: languages, tools, libraries (e.g. Python, Docker, Git, pandas)
- frameworks: web/ML (e.g. FastAPI, Django, React, scikit-learn)
- databases: database engines (e.g. PostgreSQL, Redis)
- cloud: AWS, GCP, Azure, etc.
- core_concepts: OOP, DSA, System Design, CI/CD, Unit Testing, Agile, SDLC
- experience: job titles, duration
- education: academic degrees, schools

Return valid JSON only:
{{
  "technical_skills": [],
  "frameworks": [],
  "databases": [],
  "cloud": [],
  "core_concepts": [],
  "experience": [],
  "education": []
}}

Document:
{document}
""")

explanation_prompt = ChatPromptTemplate.from_template("""
You are an expert technical recruiter matching a candidate's resume against job requirements. Use semantic matching, equivalent skills, and conceptual understanding rather than strict word-for-word comparison.

Candidate Resume Skills (Relevant match):
{resume}

Job Description Requirements:
{jd}

### Evaluation & Matching Rules:
1. Implicit Knowledge:
   - Web framework knowledge (e.g. FastAPI, Django, Express.js) implies knowledge of REST APIs / web services.
   - Git implies Version Control.
   - Docker implies Containerization.
2. Core Foundations:
   - Check if basic concepts like OOP, DSA, System Design, Unit Testing, or Agile are requested by the JD. Match them if candidate has relevant stack/experience (e.g. "OOP (implied by Python developer)").
3. Equivalent Skills:
   - Treat database systems as equivalent/transferrable (e.g. MySQL, PostgreSQL, MSSQL, SQL Server). If one is required and another is on the resume, count it as a soft-match (e.g. "PostgreSQL (via MSSQL experience)"). Do not mark it as missing.
   - Treat cloud suites as equivalent/transferrable (e.g. AWS, GCP, Azure).
   - Treat frontend frameworks as equivalent/transferrable (e.g. React, Angular, Vue).
4. Compulsory vs. Nice-to-Have Skills:
   - Distinguish required/compulsory skills from optional ones (e.g., marked as "nice to have", "good to have", "plus", "bonus", "optional", "preferred").
   - If a candidate lacks a "plus" or "nice to have" skill (like "Machine learning is a plus"), do NOT list it in "missing_skills". Only list truly mandatory requirements under "missing_skills".
5. Real-world Decisions:
   - Compare the candidate's level to the job's level. If the job requires a senior developer but the candidate lacks system architecture/design skills or has junior experience, recommend "Interview (with caution)" or "Reject" instead of "Hire".
6. Match Score Calculation:
   - Calculate a match score between 0 and 100 based on:
     a. Overlap of compulsory/must-have skills (High weight).
     b. Overlap of nice-to-have/optional skills (Low weight/bonus).
     c. Semantic similarity of the overall profile matching the job requirements.

Output valid JSON only. Do not include markdown codeblocks or extra text.
{{
  "score": 85,
  "matched_skills": ["List matched skills and equivalent capabilities (e.g. 'REST APIs (via Express.js)', 'OOP (implied by Python Developer)')"],
  "missing_skills": ["List mandatory JD requirements that are completely absent and have no equivalent in the resume (Do NOT include missing optional/nice-to-have skills here)"],
  "suggestions": ["Actionable, specific career advice for the candidate to fill gaps"],
  "overall_recommendation": "Decision (e.g. Hire, Interview, or Reject) with a clear, concise justification."
}}
""")

def parse_document(text):
    if isinstance(text, str) and text.endswith(".pdf") and os.path.exists(text):
        loader = PyPDFLoader(text)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages).strip()
    return text.strip()

def parse_pdf_bytes(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)
    return "\n".join(page.page_content for page in pages).strip()

def format_skill_list(skills_list):
    items = []
    for item in skills_list:
        if isinstance(item, dict):
            items.append(", ".join(f"{k}: {v}" for k, v in item.items()))
        else:
            items.append(str(item))
    return ", ".join(items)

def flatten_skills(skills_dict):
    parts = []
    for key in ["technical_skills", "frameworks", "databases", "cloud", "core_concepts"]:
        val = skills_dict.get(key, [])
        if val:
            parts.append(format_skill_list(val))
    return ", ".join(parts)

def get_similarity_context(resume_skills, jd_skills):
    resume_chunks = []
    for key, val in resume_skills.items():
        if val:
            resume_chunks.append(f"{key.replace('_', ' ').title()}: {format_skill_list(val)}")
    if not resume_chunks:
        return ""
    jd_full = flatten_skills(jd_skills)
    db = Chroma.from_texts(resume_chunks, embedding=embeddings_model)
    retriever = db.as_retriever(search_kwargs={"k": min(3, len(resume_chunks))})
    docs = retriever.invoke(jd_full)
    return "\n".join(d.page_content for d in docs)

def run_matching_pipeline(resume_raw, jd_raw):
    resume_text = parse_document(resume_raw)
    jd_text = parse_document(jd_raw)
    parser = JsonOutputParser()
    
    resume_msgs = skill_prompt.format_messages(document=resume_text)
    resume_resp = llm.invoke(resume_msgs)
    resume_skills = parser.invoke(resume_resp)
    
    jd_msgs = skill_prompt.format_messages(document=jd_text)
    jd_resp = llm.invoke(jd_msgs)
    jd_skills = parser.invoke(jd_resp)
    
    similarity_context = get_similarity_context(resume_skills, jd_skills)
    
    explanation_msgs = explanation_prompt.format_messages(
        resume=similarity_context,
        jd=jd_text
    )
    explanation_resp = llm.invoke(explanation_msgs)
    result = parser.invoke(explanation_resp)
    
    score = result.get("score", 0)
    return {
        "score": score,
        "similarity": round(score / 100, 4),
        "matched_skills": result.get("matched_skills", []),
        "missing_skills": result.get("missing_skills", []),
        "suggestions": result.get("suggestions", []),
        "overall_recommendation": result.get("overall_recommendation", "")
    }

class PipelineWrapper:
    def invoke(self, data):
        return run_matching_pipeline(data["resume"], data["jd"])

pipeline = PipelineWrapper()
