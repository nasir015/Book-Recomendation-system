from pickle import dump, load
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.common import save_object, load_object
from pipeline.Exception import CustomException
from pipeline.logging import logger

path = open("Log\\Model_building.txt", "w")
log_path= 'Log\\Model_building.txt'

# load the dataset
logger(log_path, "Reading the data")
df = pd.read_csv('DATA\\books_final.csv')


# preprocessing text data
logger(log_path, "preprocessing text data")
vectorizer_title = TfidfVectorizer()
tfidf_title = vectorizer_title.fit_transform(df["Title"])
logger(log_path, "preprocessing text data completed")


# save the model
logger(log_path, "saving the model")
save_object('MODELS\\vectorizer_title.pkl', vectorizer_title)
save_object('MODELS\\tfidf_title.pkl', tfidf_title)
logger(log_path, "saving the model completed")


# load the model
logger(log_path, "loading the model")
vectorizer_title = load_object('MODELS\\vectorizer_title.pkl')
tfidf_title = load_object('MODELS\\tfidf_title.pkl')
logger(log_path, "loading the model completed")


# model building
logger(log_path, "model building started")
def top_5_similar_title_books():
    
    query = str(input("Enter the book name: "))
    query = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer_title.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = df.iloc[indices]
    return results[["Title","Author","Genre","Publisher","SubGenre"]].reset_index(drop=True)

d = top_5_similar_title_books()

print(d)

logger(log_path, "model building completed")