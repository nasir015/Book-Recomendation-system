import streamlit as st
import pickle
import pandas as pd
from pickle import dump, load
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.common import save_object, load_object
from src.pipeline.Exception import CustomException
from src.pipeline.logging import logger


# load the dataset

df = pd.read_csv('DATA\\books_final.csv')


# preprocessing text data

vectorizer_title = TfidfVectorizer()
tfidf_title = vectorizer_title.fit_transform(df["Title"])



# load the model

vectorizer_title = load_object('MODELS\\vectorizer_title.pkl')
tfidf_title = load_object('MODELS\\tfidf_title.pkl')



# model building
def top_5_similar_title_books(Title):
    
    query = Title
    query = re.sub("[^a-zA-Z0-9 ]", "", query.lower())
    query_vec = vectorizer_title.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = df.iloc[indices]
    return results[["Title","Author","Genre","Publisher","SubGenre"]].reset_index(drop=True)


st.title('Book Recommendation System')

st.subheader('This is a Book Recommendation System. You can search your book and get the top 5 similar books.')

Selected_movie = st.selectbox(
    'Search Your Book',
    df['Title'].values)

if st.button('Recommend'):
    result = top_5_similar_title_books(Selected_movie)
    st.write(result)



