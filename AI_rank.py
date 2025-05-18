import streamlit as st
import extraction
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

def rank_resumes(df1, job_description):
    resume_texts = df1['Resume Text'].tolist()
    file_names = df1['File Name'].tolist()

    prompt = (
        "You are an expert HR assistant and recruiter with deep knowledge of resume screening. "
        "Your job is to strictly evaluate how well each resume matches the job description below. "
        "Be critical and objective — prioritize resumes that are highly relevant and penalize irrelevant ones. "
        "Avoid being lenient or giving equal importance to all resumes.\n\n"
        f"---\nJob Description:\n{job_description.strip()}\n---\n\n"
        "Now evaluate and rank the resumes from best fit to least fit based **only** on the job description above. "
        "Justify rankings briefly if needed. Here are the resumes:\n\n"
    )

    for i, resume in enumerate(resume_texts, 1):
        prompt += f"Resume {i}:\n{resume.strip()}\n\n"

    prompt += (
    "Return the final result in the exact format below:\n"
    "1. Resume 3 - Score: High - Strong and precise match on all required skills and relevant experience\n"
    "2. Resume 1 - Score: Medium - Partial match with some relevant skills but missing critical requirements\n"
    "3. Resume 2 - Score: Low - Does not meet the core criteria or relevant experience\n\n"
    "Only assign a 'High' score if the resume clearly demonstrates key skills and relevant experience strictly matching the job description.\n"
    "Assign 'Medium' if some relevant skills are present but important ones are missing.\n"
    "Assign 'Low' if the resume lacks critical skills or is largely irrelevant.\n"
    "Now provide your ranking in the same format. Format it **exactly** as shown above — no headings, no markdown, no extra notes."
)

    with st.spinner("🔎 Ranking resumes using Gemini..."):
        try:
            response = model.generate_content(prompt)
            output_text = response.text.strip()

            st.markdown("### 🏆 Resume Ranking Result")
            st.markdown("#### 📄 Raw Output")
            # st.code(output_text)

            # Show prompt for debugging
            # with st.expander("🧠 Prompt used", expanded=False):
            #     st.code(prompt)

            # Adjusted regex pattern (parentheses around score removed from prompt)
            pattern = r"(\d+)\.\s*Resume\s*(\d+)\s*-\s*Score:\s*(High|Medium|Low)\s*-\s*(.*)"

            matches = re.findall(pattern, output_text)

            if matches:
                parsed_data = []
                for rank, resume_num, score, reason in matches:
                    idx = int(resume_num) - 1
                    if 0 <= idx < len(file_names):
                        parsed_data.append({
                            "Rank": int(rank),
                            "File Name": file_names[idx],
                            "Score": score,
                            "Reason": reason.strip(),
                        })

                result_df = pd.DataFrame(parsed_data).sort_values(by="Rank")
                st.markdown("#### 📊 Structured Ranking")
                st.dataframe(result_df.reset_index(drop=True))
            else:
                st.warning("⚠️ Could not parse the output automatically. Please check the format or response.")
                st.text_area("Raw Gemini Output", output_text, height=300)

        except Exception as e:
            st.error(f"❌ An error occurred during resume ranking:\n\n{e}")

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
