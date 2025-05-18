from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def train_word2vec(corpus, vector_size=100, window=5, min_count=5, workers=4, sg=1, epochs=5, alpha=0.01, min_alpha=0.0001, sample=0.005, negative=3):
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        alpha=alpha,
        min_alpha=min_alpha,
        sample=sample,
        negative=negative
    )
    
    return model


def get_average_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if no word in the tokens exists in the vocabulary
        return np.zeros(model.vector_size)

    
def similarity(job_vector, resume_vectors):
    similarities = []
    for resume_vector in resume_vectors:
        similarity_score = cosine_similarity([job_vector], [resume_vector])[0][0]
        similarities.append(similarity_score)
    
    return similarities