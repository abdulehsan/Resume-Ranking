# word2vecmodel.py

from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec
from gensim.models import Word2Vec

def train_word2vec(corpus, vector_size=100, window=5, min_count=5, workers=4, sg=1, epochs=5, alpha=0.01, min_alpha=0.0001, sample=0.005, negative=3):
    """
    Train a Word2Vec model using the given corpus and parameters.

    Parameters:
    - corpus: List of tokenized sentences (list of lists of words).
    - vector_size: The dimensionality of the word vectors (default is 100).
    - window: The maximum distance between the current and predicted word within a sentence (default is 5).
    - min_count: Ignores all words with total frequency lower than this (default is 5).
    - workers: The number of CPU cores to use for training (default is 4).
    - sg: If 1, uses the skip-gram model; if 0, uses the CBOW model (default is 1).
    - epochs: The number of times to iterate over the corpus (default is 5).
    - alpha: The initial learning rate (default is 0.01).
    - min_alpha: The minimum learning rate (default is 0.0001).
    - sample: The threshold for downsampling frequent words (default is 0.005).
    - negative: The number of negative samples to use (default is 3).

    Returns:
    - model: The trained Word2Vec model.
    """
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
    """
    Given a list of tokens, return the average vector representation.
    Tokens should be a list of words, and the model should be the trained Word2Vec model.
    """
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if no word in the tokens exists in the vocabulary
        return np.zeros(model.vector_size)
