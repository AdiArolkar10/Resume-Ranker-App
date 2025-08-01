import streamlit as st
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io

def extract_text_from_pdf(pdf_file):
    # Read uploaded file into a buffer
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    # Use PyMuPDF to extract text
    text = ""
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text
  
def rank_resumes(resume_texts, job_description):
    corpus = [job_description] + resume_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity_scores

st.set_page_config(page_title="Resume Ranker", layout="wide")
st.title("📄 Resume Scanner & Ranker")

job_desc = st.text_area("Paste the Job Description", height=200)
resumes = st.file_uploader(
    "Upload Resumes (PDF only)", 
    type=["pdf"], 
    accept_multiple_files=True
)

if st.button("Rank Resumes"):
    if not job_desc or not resumes:
        st.warning("Please upload resumes and paste a job description.")
    else:
        resume_texts = []
        resume_names = []
        for file in resumes:
            text = extract_text_from_pdf(file)
            resume_texts.append(text)
            resume_names.append(file.name)

        scores = rank_resumes(resume_texts, job_desc)
        results_df = pd.DataFrame({
            'Resume': resume_names,
            'Match Score (%)': (scores * 100).round(2)
        }).sort_values(by='Match Score (%)', ascending=False)

        st.subheader("📊 Ranking Results")
        st.dataframe(results_df.reset_index(drop=True))
        st.success("Ranking Complete!")
