import tweepy
import json
import pyodbc
import re
import string
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
    with open(file) as f:
        tokens = json.load(f)

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
    text = text.lower()
    text = re.sub("RT", "", text)
    text = re.sub("rt", "", text)
    text = re.sub("@", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
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
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [lemmatizer.lemmatize(word, "v") for word in text]
    return text


def label_text(text):
    score = afn.score(text)

    if score <= -1:
        return -1
    elif score >= 1:
        return 1
    else:
        return 0


def clean_df(df):
    df["content"] = df["content"].map(lambda x: clean_text(x))
    return df


def fecth_tweets():
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
