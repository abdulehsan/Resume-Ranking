import streamlit as st
import extraction
import applyprocessing as ap
import bma
import querypre
import word2vec as w2v
from sklearn.metrics.pairwise import cosine_similarity
import explainwithllm as explain
import bert

# st.set_page_config(
#     page_title="Resume Ranking-Using BM25",
#     page_icon="📄",
#     layout="centered"
# )

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

if "ranking_done" not in st.session_state:
    st.session_state.ranking_done = False
if st.button("🔎 Rank Resumes"):
    if not uploaded_files or not job_description or not algorithms:
        st.warning("Please upload files, enter a job description, and select at least one algorithm.")
    else:
        st.session_state.ranking_done = True

        processed_text_list = preprocesseddf['Processed Text'].tolist()
        job_query = querypre.preprocess(job_description)

        for algo in algorithms:
            if algo == "BM25":
                st.subheader("📊 BM25 Ranking")
                bm25_result = bma.applybm25(processed_text_list, job_query)
                st.session_state.bm25_ranked = sorted(zip(bm25_result, df1['File Name']), reverse=True)
                st.write("### 📈 Ranked Resumes (BM25)")
                for i, (score, file_name) in enumerate(st.session_state.bm25_ranked, start=1):
                    st.markdown(f"**{i}. {file_name}** — Similarity Score: `{score:.4f}`")

            if algo == "Word2Vec":
                st.subheader("Word2Vec Ranking")
                model = w2v.train_word2vec([tokens for tokens in processed_text_list])
                resume_vectors = [w2v.get_average_vector(tokens, model) for tokens in processed_text_list]
                job_vector = w2v.get_average_vector(job_query, model)
                similarities = [cosine_similarity([resume_vector], [job_vector])[0][0] for resume_vector in resume_vectors]
                st.session_state.word2vec_ranked = sorted(zip(similarities, df1['File Name']), reverse=True)
                st.write("### 📈 Ranked Resumes (Word2Vec)")
                for i, (score, file_name) in enumerate(st.session_state.word2vec_ranked, start=1):
                    st.markdown(f"**{i}. {file_name}** — Similarity Score: `{score:.4f}`")
                    
            elif algo == "BERT":
                st.subheader("BERT Ranking")
                similarities = bert.applybert(df1["Resume Text"].tolist(), job_description)
                st.session_state.bert_ranked = sorted(zip(similarities, df1['File Name']), reverse=True)
                st.write("### 📈 Ranked Resumes (BERT)")
                for i, (score, file_name) in enumerate(st.session_state.bert_ranked, start=1):
                    st.markdown(f"**{i}. {file_name}** — Similarity Score: `{score:.4f}`")
        
if st.session_state.get("ranking_done", False):
    for algo in algorithms:
        algo_key = algo.lower().replace("-", "").replace(" ", "")
        rank_key = f"{algo_key}_ranked"

        if rank_key in st.session_state:
            st.markdown(f"---\n## 🤖 Groq Evaluation for {algo}")
            resume_score_data = [
                (file_name, score, preprocesseddf[preprocesseddf['File Name'] == file_name]['Processed Text'].values[0])
                for score, file_name in st.session_state[rank_key]
            ]

            if st.button(f"Use Groq to Classify & Explain ({algo})"):
                good_fits, bad_fits = explain.  batch_groq_fit_evaluation(algo, job_description, resume_score_data)

                st.subheader("✅ Good Fit Resumes")
                for file_name, score, explanation in good_fits:
                    st.markdown(f"**{file_name}** — Score: `{score:.4f}`")
                    st.markdown(explanation)
                    st.markdown("---")

                st.subheader("❌ Bad Fit Resumes")
                for file_name, score, explanation in bad_fits:
                    st.markdown(f"**{file_name}** — Score: `{score:.4f}`")
                    st.markdown(explanation)
                    st.markdown("---")
