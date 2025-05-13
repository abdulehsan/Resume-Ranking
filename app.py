import streamlit as st
import extraction
import applyprocessing as ap
import bma
import querypre
import word2vec as w2v
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import google.generativeai as genai
import requests

# Hardcoding your Gemini API key
genai.configure(api_key="AIzaSyAlNq-uhihqv0FSpgD7Jdo5sDIX0xeWtGQ")
def explain_with_groq(algorithm_name, job_description, resume_text, score):
    prompt = f"""
    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Similarity Score: {score:.4f}

    Explain in 3-5 bullet points why this resume is considered highly relevant using {algorithm_name}.
    """
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error using Groq API: {e}"

st.set_page_config(
    page_title="Resume Ranking-Using BM25",
    page_icon="📄",
    layout="centered"
)

st.title("Resume Ranking-Using BM25")
st.subheader("Upload your resumes to rank them using BM25, etc.")
st.markdown("Start uploading resumes and a job description below:")

uploaded_files = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded")
    
    df1 = extraction.extract_text(uploaded_files)
    
    st.dataframe(df1[['File Name', 'Resume Text']])
    
    preprocesseddf = ap.preprocess_resumes(df1)
    for index, row in preprocesseddf.iterrows():
        with st.expander(f"Resume {row['File Name']}"):
            st.write(row['Processed Text'])

            # ✅ GEMINI RAG EXPLANATION SECTION — INSIDE THIS BLOCK
    GROQ_API_KEY = "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"

    selected_resume = st.selectbox("Select a Resume", df1['File Name'].tolist())
    job_description = st.text_area("Paste the Job Description", key="job_description_1")

    if st.button("Generate Explanation with Mixtral"):
        selected_text = preprocesseddf[preprocesseddf['File Name'] == selected_resume]['Processed Text'].values[0]
        
        prompt = f"""
        Job Description:
        {job_description}

        Resume:
        {selected_text}

        Based on the job description and resume above, explain in 3-5 bullet points why this resume may or may not be a good fit.
        """

        with st.spinner("Calling Mixtral via Groq..."):
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
                response = requests.post(url, headers=headers, json=payload)
                result = response.json()
                print(result)  # 👈 Add this line
                explanation = result['choices'][0]['message']['content']
                st.success("Explanation Generated")
                st.markdown(explanation)
            except Exception as e:
                st.error(f"Error using Groq API: {e}")
else:
    st.info("You can upload multiple PDF, DOC files.")

job_description = st.text_area(
    "Paste the Job Description Here",
    placeholder="Example: We're looking for a Python developer with experience in NLP and machine learning..."
)

st.markdown("---")

algorithms = st.multiselect(
    "Select Ranking Algorithm(s)",
    ["TF-IDF", "BM25", "Word2Vec", "BERT"],
    default=["TF-IDF"]
)

with st.expander("ℹ️ What do these algorithms mean?"):
    st.markdown(""" 
    - **TF-IDF**: Calculates word importance by frequency.
    - **BM25**: A probabilistic ranking method.
    - **Word2Vec**: Uses word embeddings for context.
    - **BERT**: Deep contextualized transformer model.
    """)

if st.button("🔎 Rank Resumes"):
    if not uploaded_files or not job_description or not algorithms:
        st.warning("Please upload files, enter a job description, and select at least one algorithm.")
    else:
        st.info("Processing resumes... (this may take a few seconds)")
        
        # Preprocessed resume texts
        processed_text_list = preprocesseddf['Processed Text'].tolist()  # List of tokenized resumes
        
        # Tokenize the job description
        job_query = querypre.preprocess(job_description)  # Tokenized job description
        
        # BM25 Ranking
        if "BM25" in algorithms:
            st.subheader("📊 BM25 Ranking")
            bm25_result = bma.applybm25(processed_text_list, job_query)
            st.write(bm25_result)
        
        # Word2Vec Ranking
        if "Word2Vec" in algorithms:
            st.subheader("Word2Vec Ranking")

            # Train model
            model = w2v.train_word2vec([tokens for tokens in processed_text_list])  
            resume_vectors = [w2v.get_average_vector(tokens, model) for tokens in processed_text_list]
            job_vector = w2v.get_average_vector(job_query, model)  

            similarities = [cosine_similarity([resume_vector], [job_vector])[0][0] for resume_vector in resume_vectors]
            st.session_state.word2vec_ranked = sorted(zip(similarities, df1['File Name']), reverse=True)
            st.write("### 📈 Ranked Resumes (Word2Vec)")
            for i, (score, file_name) in enumerate(st.session_state.word2vec_ranked, start=1):
                st.markdown(f"**{i}. {file_name}** — Similarity Score: `{score:.4f}`")


if "word2vec_ranked" in st.session_state:
    st.subheader("🔍 Explain a Resume Match (Word2Vec + Groq)")
    
    selected_resume_word2vec = st.selectbox(
        "Select a Resume (Word2Vec)", 
        [file_name for _, file_name in st.session_state.word2vec_ranked], 
        key="word2vec_resume_select"
    )

    if st.button("Ask Groq for Explanation (Word2Vec)"):
        selected_text = preprocesseddf[preprocesseddf['File Name'] == selected_resume_word2vec]['Processed Text'].values[0]
        selected_score = next(
    (score for score, file_name in st.session_state.word2vec_ranked if file_name == selected_resume_word2vec),
    None
)

        prompt = f"""
        Job Description:
        {job_description}

        Resume:
        {selected_text}

        Similarity Score: {selected_score:.4f}

        Explain in 3-5 bullet points why this resume is a good fit using Word2Vec.
        """

        with st.spinner("Asking Groq..."):
            explanation = explain_with_groq("Word2Vec", job_description, selected_text, selected_score)
            st.success("Explanation Generated")
            st.markdown(explanation)


