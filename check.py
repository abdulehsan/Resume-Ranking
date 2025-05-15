import streamlit as st
import extraction
import applyprocessing as ap
import bma
import querypre
from word2vec import train_word2vec, get_average_vector, similarity  # Import from word2vecmodel.py
import os
import google.generativeai as genai
import requests
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("API_KEY")
# Hardcoding your Gemini API key
genai.configure(api_key="AIzaSyAlNq-uhihqv0FSpgD7Jdo5sDIX0xeWtGQ")

def explain_with_groq(algorithm_name, job_description, resume_text, score):
    prompt = f"""
    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Similarity Score: {score:.4f}

    Explain in 3-5 bullet points why this resume is considered relevant using {algorithm_name}.
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
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "Error: 'choices' not found in the response."
    except Exception as e:
        return f"Error using Groq API: {e}"

st.set_page_config(
    page_title="Resume Ranking-Using BM25",
    page_icon="📄",
    layout="centered"
)

st.title("Resume Ranking")
st.subheader("Upload your resumes to rank them using BM25, Word2Vec, etc.")
st.markdown("Start uploading resumes and a job description below:")

uploaded_files = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    st.success(f"{len(uploaded_files)} files uploaded")
    
    df1 = extraction.extract_text(uploaded_files)
    
    st.dataframe(df1[['File Name', 'Resume Text']])
    
    preprocesseddf = ap.preprocess_resumes(df1)
    job_description = st.text_area("Paste the Job Description", key="job_description_1")

else:
    st.info("You can upload multiple PDF, DOC files.")

st.markdown("---")

algorithms = st.selectbox(
    "Select Ranking Algorithm(s)",
    ["TF-IDF", "BM25", "Word2Vec", "BERT"],
)

# with st.expander("ℹ️ What do these algorithms mean?"):
#     st.markdown(""" 
#     - **TF-IDF**: Calculates word importance by frequency.
#     - **BM25**: A probabilistic ranking method.
#     - **Word2Vec**: Uses word embeddings for context.
#     - **BERT**: Deep contextualized transformer model.
#     """)

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
            ranked_resumes = sorted(zip(bm25_result, df1['File Name']), reverse=True)
            st.session_state.rankings = ranked_resumes  # Store the rankings in session state
            for i, (score, file_name) in enumerate(ranked_resumes, start=1):
                st.markdown(f"**{i}. {file_name}** — Similarity Score: `{score:.4f}`")
            st.write(bm25_result)

        # Word2Vec Ranking
        if "Word2Vec" in algorithms:
            st.subheader("Word2Vec Ranking")

            # Train Word2Vec model
            model = train_word2vec([tokens for tokens in processed_text_list])  # Pass the tokenized sentences
            resume_vectors = [get_average_vector(tokens, model) for tokens in processed_text_list]
            job_vector = get_average_vector(job_query, model)

            similarities = similarity(job_vector, resume_vectors)
            ranked_resumes = sorted(zip(similarities, df1['File Name']), reverse=True)

            st.session_state.rankings = ranked_resumes  # Store the rankings in session state

            st.write("### 📈 Ranked Resumes (Word2Vec)")
            for i, (score, file_name) in enumerate(ranked_resumes, start=1):
                st.markdown(f"**{i}. {file_name}** — Similarity Score: `{score:.4f}`")

if "rankings" in st.session_state:
    st.subheader("🤖 LLM Explanation for Ranked Resume")

    selected_resume = st.selectbox(
        "Select a Resume from Word2Vec Ranking",
        [file_name for _, file_name in st.session_state.rankings],
        key="explain_resume_choice"
    )

    if st.button("Ask Groq for Explanation"):
        selected_text = preprocesseddf[preprocesseddf['File Name'] == selected_resume]['Processed Text'].values[0]
        selected_score = next(
            (score for score, file_name in st.session_state.rankings if file_name == selected_resume),
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
            explanation = explain_with_groq(algorithms, job_description, selected_text, selected_score)
            st.success("Explanation Generated")
            st.markdown(explanation)
