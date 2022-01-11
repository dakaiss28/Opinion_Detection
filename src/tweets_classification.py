""" this module connects to the twitter API, get tweets, clean and labelize them
and them store them in a dataBase"""
import itertools
import json
import re
import string
import pyodbc
from sklearn.utils import shuffle
import tweepy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from wordcloud import WordCloud

brands = ["nigeria"]
stop_words = stopwords.words("english") + brands
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


def plot_word_cloud(text, fichier):
    """plot word could"""
    wc = WordCloud(
        width=600,
        height=600,
        background_color="white",
        max_words=200,
        stopwords=stop_words,
        max_font_size=90,
        collocations=False,
        random_state=42,
    )

    # Générer et afficher le nuage de mots
    plt.figure(figsize=(15, 20))
    wc.generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    plt.savefig(fichier)


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
        for tweet in tweepy.Cursor(
            api.search_tweets, lang="en", q=brand, result_type="recent"
        ).items(6000):
            print(tweet.text)
            label = input("Label : ")
            cursor.execute(
                "INSERT INTO dbo.tweets values(?,?,?,?,?,?,?,?,?)",
                tweet.id_str,
                tweet.created_at,
                tweet.text,
                int(tweet.retweet_count),
                int(tweet.favorite_count),
                brand,
                int(label),
                -1,
                -1,
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


def train_word2vec(tweets_df):
    """transform the texts into vector using word2vec model"""
    data = tweets_df["content"].map(lambda x: x.split(" ")).tolist()
    model = Word2Vec(sentences=data, vector_size=100, workers=1, seed=1)
    features = []

    for tokens in data:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


def classify_tweets_svm(tweets_df):
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(
        tweets_df["content"], tweets_df["target_label"], test_size=0.3
    )
    SVM = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(tweets_df["content"])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    SVM.fit(Train_X_Tfidf, Train_Y)  # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)  # Use accuracy_score function to get the accuracy

    accuracy = accuracy_score(Test_Y, predictions_SVM)
    precision = precision_score(Test_Y, predictions_SVM, average=None)
    recall = recall_score(Test_Y, predictions_SVM, average=None)
    sns.countplot(Test_Y)
    plt.savefig("../plots/SVMdistributionTrue.png")
    sns.countplot(predictions_SVM)
    plt.savefig("../plots/SVMdistributionPred.png")
    print("accuracy for SVM classifier : {}".format(accuracy))
    print("precision for SVM classifier : {}".format(precision))
    print("recall for SVM classifier : {}".format(recall))


def classify_tweets_kmeans(features, tweets_df):
    """This method convert the twwets text into vectors and
    then classify them using K Means classifier"""

    kmeans_classifier = KMeans(n_clusters=3, max_iter=600, random_state=1)
    pca = PCA(n_components=2)

    tfidf_vect = TfidfVectorizer(max_features=5000)
    tf_idf = tfidf_vect.fit_transform(tweets_df["content"])
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()
    y_learn = pca.fit_transform(tf_idf_array)
    fitted = kmeans_classifier.fit(features)
    prediction = kmeans_classifier.predict(features)

    return (y_learn, fitted, prediction, tf_idf_array, tfidf_vect)


def plot_kmean_results(tweets_df, y_learn, fitted, prediction, tf_idf_array, tfidf_vect):
    res = []
    for id, result in itertools.zip_longest(tweets_df["tweet_id"], fitted.labels_):
        res.append({"id": id, "label": result})
    results = pd.DataFrame(res, columns=["id", "label"])
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
        plt.savefig("../plots/frequentWords{}.png".format(label))
    plt.show()
    return results


def get_top_features_cluster(tf_idf_array, prediction, label, tfidf_vect, n_feats):
    """get top features per cluster"""
    id_temp = np.where(prediction == label)  # indices for each cluster
    x_means = np.mean(tf_idf_array[id_temp], axis=0)  # returns average score across cluster
    sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
    features = tfidf_vect.get_feature_names_out()
    best_features = [(features[i], x_means[i]) for i in sorted_means]
    return pd.DataFrame(best_features, columns=["features", "score"])


def store_results(result, conf, model):
    """store kmeans results in dataBase table"""
    (_, connexion) = set_up(conf)
    cursor = connexion.cursor()
    for _, row in result.iterrows():
        cursor.execute(
            "UPDATE dbo.tweets SET " + model + " = ? WHERE tweet_id = ?",
            row["label"],
            row["id"],
        )
    connexion.commit()
    connexion.close()


def temporal_evolution(conf):
    """plot the temporal evolution of labels"""
    (_, connexion) = set_up(conf)
    results = pd.read_sql("SELECT created_at,kmeans_res FROM dbo.tweets", connexion)
    # results["created_at"].apply(lambda x: x.timestamp())

    sns.countplot(x="kmeans_res", data=results).set(title="Label repartition ")
    plt.savefig("../plots/distributionCount.png")
    dates = pd.read_sql("SELECT created_at FROM dbo.tweets", connexion)

    results2 = pd.DataFrame()
    for _, row in dates.iterrows():
        results2 = results2.append(
            pd.read_sql(
                "SELECT  count(kmeans_res) as count_res, created_at, kmeans_res FROM dbo.tweets where created_at = ? group by kmeans_res,created_at ",
                connexion,
                params=[row["created_at"]],
            )
        )

    plt.show()

    for km, gp in results2.groupby("kmeans_res"):
        gp.plot(x="created_at", y="count_res", label=km)
        plt.savefig("../plots/overtimeRepartition{}.png".format(km))


def main():
    """main method"""
    # fetch_tweets("../twitter_token.json.txt")
    tweets_df = retrieve_tweets()
    data_frame = clean_df(tweets_df)
    # plot_word_cloud(str(data_frame["content"].values), "../plots/cloud.png")
    features = train_word2vec(data_frame)
    (y_learn, fitted, prediction, tf_idf_array, tfidf_vect) = classify_tweets_kmeans(
        features, data_frame
    )
    results_kmean = plot_kmean_results(
        data_frame, y_learn, fitted, prediction, tf_idf_array, tfidf_vect
    )
    store_results(results_kmean, "../twitter_token.json.txt", "kmeans_res")
    classify_tweets_svm(data_frame)

    temporal_evolution("../twitter_token.json.txt")


if __name__ == "__main__":
    main()
