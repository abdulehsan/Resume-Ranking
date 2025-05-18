import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):    
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def preprocess_resumes(df):
    df['Processed Text'] = df['Resume Text'].apply(preprocess)
    return df


import re

def preprocess_for_llm(tokens):
    text = ' '.join(tokens)

    # Remove common filler phrases or generic text patterns
    junk_phrases = [
        "curriculum vitae", "references available", "responsible for",
        "team player", "hardworking", "detail oriented", "good communication"
    ]
    for phrase in junk_phrases:
        text = text.replace(phrase, '')

    # Remove email addresses, phone numbers, and long IDs
    text = re.sub(r'\S+@\S+', '', text)  # emails
    text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}', '', text)  # phone numbers
    text = re.sub(r'\b[a-z]*\d+[a-z]*\b', '', text)  # random alphanumeric IDs

    # Remove short/meaningless tokens again
    final_tokens = [word for word in word_tokenize(text) if len(word) > 2]

    return final_tokens
