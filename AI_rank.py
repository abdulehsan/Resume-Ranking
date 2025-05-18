import streamlit as st
import extraction
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

def rank_resumes(df1, job_description):
    resume_texts = df1['Resume Text'].tolist()
    file_names = df1['File Name'].tolist()

    prompt = (
        f"You are a professional HR assistant. "
        f"Based on the following job description, rank the resumes from best fit to least fit.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resumes:\n"
    )
    for i, text in enumerate(resume_texts):
        prompt += f"\nResume {i+1} (File Name: {file_names[i]}):\n{text}\n"

    with st.spinner("Ranking resumes using AI..."):
        try:
            response = model.generate_content(prompt)
            output_text = response.text

            st.markdown("### 🏆 Resume Ranking Result")
            st.markdown("#### Raw Output")
            st.write(output_text)

            # Optional: Parse the output_text into a structured DataFrame (depends on format)
            # You can implement this if Gemini returns JSON or number-based rankings

        except Exception as e:
            st.error(f"An error occurred during resume ranking: {e}")

# Streamlit App
# st.set_page_config(page_title="Resume Ranker", layout="wide")
st.title("📄 Resume Ranking Using Gemini AI")
st.subheader("Upload resumes and a job description. Let AI rank them!")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} files uploaded.")
    df1 = extraction.extract_text(uploaded_files)
    st.dataframe(df1[['File Name', 'Resume Text']])
else:
    st.info("You can upload multiple resume files (PDF, DOCX, TXT).")

job_description = st.text_area(
    "📋 Paste the Job Description Here",
    placeholder="Example: We are looking for a software engineer with experience in NLP, Python, and TensorFlow."
)

if uploaded_files and job_description:
    st.markdown("---")
    st.subheader("🔎 Rank Resumes using AI")
    if st.button("Start Ranking"):
        rank_resumes(df1, job_description)
