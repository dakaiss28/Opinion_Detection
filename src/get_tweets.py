""" this module connects to the twitter API, get tweets, clean and labelize them
and them store them in a dataBase"""
import json
import re
import string
import pyodbc
import tweepy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from afinn import Afinn

brands = ["iPhone", "iPad", "macbook", "airPods"]
stop_words = stopwords.words("english") + brands + ["Pro", "I", "follow", "Follow"]
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
afn = Afinn()


def set_up(file):
    """setting up the application : connection to the Twitter API and the local dataBase"""
    with open(file, encoding=str) as set_up_file:
        tokens = json.load(set_up_file)

        consumer_key = tokens["Key"]
        consumer_secret = tokens["Secret"]

        auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

        api = tweepy.API(auth)

    db_connexion = pyodbc.connect(
        "Driver={SQL Server Native Client 11.0};"
        "Server=localhost;"
        "Database=tweets;"
        "Trusted_Connection=yes;"
    )

    return (api, db_connexion)


def clean_text(text):
    """cleaning tweets text"""
    text = text.lower()
    text = re.sub("RT", "", text)
    text = re.sub("rt", "", text)
    text = re.sub("@", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)

    # Tokenize the data
    text = nltk.word_tokenize(text)

    # Lemmatize text
    text = lemmatize_text(text)

    # Remove stopwords
    text = [word for word in text if word not in stop_words]
    return " ".join(text)


def lemmatize_text(text):
    "lemmatize the tweets text"
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [lemmatizer.lemmatize(word, "v") for word in text]
    return text


def label_text(text):
    "labelize the tweets using Afinn library"
    score = afn.score(text)
    final_score = 0
    if score <= -1:
        final_score = -1
    elif score >= 1:
        final_score = 1
    return final_score


def clean_df(data_frame):
    "iterate through the dataframe to clean the text"
    data_frame["content"] = data_frame["content"].map(clean_text)
    return data_frame


def fecth_tweets():
    "get the tweets from the API and store them in the dataBase"
    (api, db_connexion) = set_up("twitter_token.json.txt")
    cursor = db_connexion.cursor()
    for brand in brands:
        for tweet in tweepy.Cursor(api.search_tweets, lang="en", q=brand).items(1000):
            cursor.execute(
                "INSERT INTO dbo.tweets values(?,?,?,?,?,?,?)",
                int(tweet.id),
                tweet.created_at,
                tweet.text,
                int(tweet.retweet_count),
                int(tweet.favorite_count),
                brand,
                label_text(tweet.text),
            )
            db_connexion.commit()
    db_connexion.close()
