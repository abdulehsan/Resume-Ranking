from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def applybert(resume_texts, job_text):
    
        # Encode the job description and resumes
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = cosine_similarity(job_embedding, resume_embeddings)[0]
    return cosine_scores
    # Create a DataFrame to store results
    # results_df = pd.DataFrame({
    #     'Resume Text': resume_texts,
    #     'Similarity Score': cosine_scores.cpu().numpy()
    # })

    # # Sort by similarity score in descending order
    # results_df = results_df.sort_values(by='Similarity Score', ascending=False).reset_index(drop=True)

    # return results_df