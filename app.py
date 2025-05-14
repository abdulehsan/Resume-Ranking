import streamlit as st
import extraction
import applyprocessing as ap
import bma
import querypre
import word2vec as w2v
from sklearn.metrics.pairwise import cosine_similarity
import bert

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
else:
    st.info("You can upload multiple PDF, DOC files.")

job_description = st.text_area(
    "Paste the Job Description Here",
    placeholder="Example: We're looking for a Python developer with experience in NLP and machine learning..."
)

st.markdown("---")

algorithms = st.selectbox(
    "Select Ranking Algorithm(s)",
    ["TF-IDF", "BM25", "Word2Vec", "BERT"],
    # default=["TF-IDF"]
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
        elif "Word2Vec" in algorithms:
         st.subheader("Word2Vec Ranking")
         model = w2v.train_word2vec([tokens for tokens in processed_text_list])  
         resume_vectors = [w2v.get_average_vector(tokens, model) for tokens in processed_text_list]
         job_vector = w2v.get_average_vector(job_query, model)  
         similarities = [cosine_similarity([resume_vector], [job_vector])[0][0] for resume_vector in resume_vectors]
         ranked_resumes = sorted(zip(similarities, df1['File Name']), reverse=True)
         st.write("Ranked Resumes based on Word2Vec Similarity:")
         for score, file_name in ranked_resumes:
             st.write(f"{file_name}: {score:.4f}")
        elif "BERT" in algorithms:
            st.subheader("BERT Ranking")
            # Assuming you have a function to apply BERT ranking
            bert_result = bert.applybert( job_description,processed_text_list)
            st.write(bert_result)
     
     