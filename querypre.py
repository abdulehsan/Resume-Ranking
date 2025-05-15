import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess(job):
    job = job.lower().strip()
    tokens = word_tokenize(job)
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    return filtered


