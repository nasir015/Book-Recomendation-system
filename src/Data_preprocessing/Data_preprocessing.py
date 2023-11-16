
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re
from pickle import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.logging import logger
from pipeline.Exception import CustomException

path = open("Log\\Data_Preprocessing.txt", "w")
log_path= 'Log\\Data_Preprocessing.txt'

logger(log_path, "Data Preprocessing Started")



logger(log_path, "Reading the data")

# Reading the data
def reading_and_marge_data():
    books = pd.read_csv('Data\\books.csv')
    books_new = pd.read_csv('Data\\books_new.csv')
    df = books
    df['SubGenre'] = books_new['SubGenre']
    return df



def preprocess_dataset(df):
    logger(log_path, 'preprocess_dataset started')

    # droping the duplicates
    df.drop_duplicates(inplace=True)
    logger(log_path, 'droping the duplicates completed')


    # preprocessing numercial data
    df["Height"] = pd.to_numeric(df["Height"])
    logger(log_path, 'preprocessing numercial data completed')

    # preprocessing text data

    stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])
    
    df["Title"] = df["Title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["Title"] = df["Title"].astype(str).apply(lambda sentence: ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords and len(e.lower())>1))
    df["Title"] = df["Title"].str.replace("\s+", " ", regex=True)
    

    df["Author"] = df["Author"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["Author"] = df["Author"].str.lower()
    df["Author"] = df["Author"].str.replace("\s+", " ", regex=True)
    


    df["Genre"] = df["Genre"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["Genre"] = df["Genre"].str.lower()
    df["Genre"] = df["Genre"].str.replace("\s+", " ", regex=True)
    

    df["Publisher"] = df["Publisher"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["Publisher"] = df["Publisher"].str.lower()
    df["Publisher"] = df["Publisher"].str.replace("\s+", " ", regex=True)
    


    df["SubGenre"] = df["SubGenre"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
    df["SubGenre"] = df["SubGenre"].str.lower()
    df["SubGenre"] = df["SubGenre"].str.replace("\s+", " ", regex=True)
    
    logger(log_path, 'preprocessing text data completed')

    # save the dataframe as csv file
    df.to_csv('Data\\books_final.csv', index=False)

    logger(log_path, 'Dataframe saved as csv file')



if __name__ == "__main__":
    df = reading_and_marge_data()
    preprocess_dataset(df)
    