import streamlit as st
import PyPDF2
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.title("🚀 Advanced AI Resume ATS Analyzer")

# File upload
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here")


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors)[0][1]
    return round(similarity * 100, 2)


def extract_keywords(text):
    text = text.lower()

    skills = [
        "python", "sql", "excel", "power bi", "tableau",
        "machine learning", "data visualization",
        "data cleaning", "scikit-learn",
        "pandas", "numpy", "statistics",
        "communication skills", "dashboards",
        "reports", "stakeholders"
    ]

    return [skill for skill in skills if skill in text]


def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def check_action_verbs(text):
    action_verbs = ["developed", "implemented", "designed", "built", "analyzed", "optimized"]
    return [verb for verb in action_verbs if verb in text.lower()]


def check_quantified_results(text):
    return re.findall(r'\d+%|\d+\+', text)


# 🚀 MAIN EXECUTION BLOCK (VERY IMPORTANT)
if uploaded_file is not None and job_description.strip() != "":

    resume_text = extract_text_from_pdf(uploaded_file)

    score = calculate_similarity(
        resume_text.lower(),
        job_description.lower()
    )

    # Score Display
    st.subheader(f"ATS Match Score: {score}%")

    if score > 80:
        st.success("Excellent Match 🔥")
    elif score > 60:
        st.warning("Moderate Match ⚠ Improve Keywords")
    else:
        st.error("Low Match ❌ Add More Relevant Skills")

    # Keyword Comparison
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)

    missing_keywords = list(set(jd_keywords) - set(resume_keywords))

    st.subheader("📌 Missing Keywords")
    st.write(missing_keywords)

    # NLP Insights
    st.subheader("📍 Named Entities in Resume")
    st.write(extract_entities(resume_text))

    # Resume Strength
    st.subheader("💪 Resume Strength Analysis")
    action_verbs_found = check_action_verbs(resume_text)
    quantified_results = check_quantified_results(resume_text)

    st.write("Action Verbs Used:", action_verbs_found)
    st.write("Quantified Achievements Found:", quantified_results)

    # Suggestions
    st.subheader("🧠 Smart Suggestions")

    if len(action_verbs_found) < 3:
        st.write("Add more strong action verbs like: Developed, Led, Improved")

    if len(quantified_results) == 0:
        st.write("Add measurable achievements (e.g., Increased sales by 30%)")

    if len(missing_keywords) > 5:
        st.write("Resume missing many job-relevant keywords. Improve skill alignment.")

else:
    st.info("Please upload resume and paste job description.")
