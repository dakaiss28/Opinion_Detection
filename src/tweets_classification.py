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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

brands = ["iPhone", "iPad", "macbook", "airPods"]
stop_words = stopwords.words("english") + brands + ["Pro", "I", "follow", "Follow"]
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()


def set_up(conf):
    """setting up the application : connection to the Twitter API and the local dataBase"""
    with open(conf) as set_up_file:
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


def clean_df(data_frame):
    "iterate through the dataframe to clean the text"
    data_frame["content"] = data_frame["content"].map(clean_text)
    return data_frame


def fetch_tweets(conf):
    "get the tweets from the API and store them in the dataBase"
    (api, db_connexion) = set_up(conf)
    cursor = db_connexion.cursor()
    for brand in brands:
        for tweet in tweepy.Cursor(api.search_tweets, lang="en", q=brand).items(1000):
            cursor.execute(
                "INSERT INTO dbo.tweets values(?,?,?,?,?,?)",
                int(tweet.id),
                tweet.created_at,
                tweet.text,
                int(tweet.retweet_count),
                int(tweet.favorite_count),
                brand,
            )
            db_connexion.commit()
    db_connexion.close()


def retrieve_tweets():
    """retrieve tweets from Twitter"""
    db_connexion = pyodbc.connect(
        "Driver={SQL Server Native Client 11.0};"
        "Server=localhost;"
        "Database=tweets;"
        "Trusted_Connection=yes;"
    )

    query = "SELECT * FROM dbo.tweets"
    tweets_df = pd.read_sql(query, db_connexion)
    tweets_df.drop_duplicates()
    db_connexion.close()
    return tweets_df


def classify_tweets_kmeans(tweets_df):
    """This method convert the twwets text into vectors and
    then classify them using K Means classifier"""

    kmeans_classifier = KMeans(n_clusters=3, max_iter=600, algorithm="full")
    pca = PCA(n_components=2)

    tfidf_vect = TfidfVectorizer(max_features=5000)
    tf_idf = tfidf_vect.fit_transform(tweets_df["content"])
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()

    y_learn = pca.fit_transform(tf_idf_array)
    fitted = kmeans_classifier.fit(y_learn)
    prediction = kmeans_classifier.predict(y_learn)

    plt.scatter(y_learn[:, 0], y_learn[:, 1], c=prediction, s=50, cmap="viridis")
    centers = fitted.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=300, alpha=0.6)
    # plt.show()
    plt.savefig("../plots/labelDistribution.png")
    plt.figure(figsize=(8, 6))
    labels = np.unique(prediction)
    for label in labels:
        dfs = get_top_features_cluster(tf_idf_array, prediction, label, tfidf_vect, 15)
        sns.barplot(x="score", y="features", orient="h", data=dfs).set(
            title="top features for class {}".format(label)
        )
        # plt.show()
        plt.savefig("../plots/frequentWords{}.png".format(label))


def get_top_features_cluster(tf_idf_array, prediction, label, tfidf_vect, n_feats):
    """get top features per cluster"""
    id_temp = np.where(prediction == label)  # indices for each cluster
    x_means = np.mean(
        tf_idf_array[id_temp], axis=0
    )  # returns average score across cluster
    sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
    features = tfidf_vect.get_feature_names_out()
    best_features = [(features[i], x_means[i]) for i in sorted_means]
    return pd.DataFrame(best_features, columns=["features", "score"])


def main():
    """main method"""
    # fetch_tweets("../twitter_token.json.txt")
    tweets_df = retrieve_tweets()
    data_frame = clean_df(tweets_df)
    classify_tweets_kmeans(data_frame)


if __name__ == "__main__":
    main()
