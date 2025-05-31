from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def applybert(resume_texts, job_text):
    resume_embeddings = model.encode(resume_texts, convert_to_numpy=True)
    
    job_embedding = model.encode(job_text, convert_to_numpy=True).reshape(1, -1)
    
    cosine_scores = cosine_similarity(resume_embeddings, job_embedding).flatten()
    
    return cosine_scores
