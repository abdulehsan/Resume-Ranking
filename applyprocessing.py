import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

stemmer = PorterStemmer()

def preprocess(text):    
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]    
    return tokens

def preprocess_resumes(df):
    df['Processed Text'] = df['Resume Text'].apply(preprocess)
    return df
