from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def applybert(resume_texts, job_text):
    # Encode all resumes at once (returns tensor)
    resume_embeddings = model.encode(resume_texts, convert_to_numpy=True)
    
    # Encode job description (as 2D array for cosine similarity)
    job_embedding = model.encode(job_text, convert_to_numpy=True).reshape(1, -1)
    
    # Calculate cosine similarity between job and all resumes
    cosine_scores = cosine_similarity(resume_embeddings, job_embedding).flatten()
    
    return cosine_scores
