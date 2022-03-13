""" this module connects to the twitter API, get tweets, clean and labelize them
and then store them in a dataBase"""
import itertools
import re
import string
from sklearn.utils import shuffle
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
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from utils import *
from wordcloud import WordCloud

stop_words = stopwords.words("english") + brands
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()


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
    "this methods classifies tweets using SVM method"
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(
        tweets_df["content"], tweets_df["target_label"], test_size=0.3
    )
    SVM = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(tweets_df["content"])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    SVM.fit(Train_X_Tfidf, Train_Y)  # predict the labels on validation dataset
    predictions_SVM = SVM.predict(
        Test_X_Tfidf
    )  # Use accuracy_score function to get the accuracy

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


def plot_kmean_results(
    tweets_df, y_learn, fitted, prediction, tf_idf_array, tfidf_vect
):
    """plots kmean results : the label distribution and frequent words for each class"""
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
    x_means = np.mean(
        tf_idf_array[id_temp], axis=0
    )  # returns average score across cluster
    sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
    features = tfidf_vect.get_feature_names_out()
    best_features = [(features[i], x_means[i]) for i in sorted_means]
    return pd.DataFrame(best_features, columns=["features", "score"])


def get_label_repartition_kmeans(connexion):
    "get kmeans label distribution"
    query = "SELECT created_at,kmeans_res FROM dbo.tweets"
    results = pd.read_sql(query, connexion)

    sns.countplot(x="kmeans_res", data=results).set(title="Label repartition ")
    plt.savefig("../plots/distributionCount.png")


def main():
    """main method"""
    _, db_connexion = set_up("../twitter_token.json.txt")
    tweets_df = retrieve_tweets()
    data_frame = clean_df(tweets_df)
    plot_word_cloud(str(data_frame["content"].values), "../plots/cloud.png")
    features = train_word2vec(data_frame)
    (y_learn, fitted, prediction, tf_idf_array, tfidf_vect) = classify_tweets_kmeans(
        features, data_frame
    )
    results_kmean = plot_kmean_results(
        data_frame, y_learn, fitted, prediction, tf_idf_array, tfidf_vect
    )
    store_results(results_kmean, "../twitter_token.json.txt", "kmeans_res")
    classify_tweets_svm(data_frame)
    get_label_repartition_kmeans(db_connexion)


if __name__ == "__main__":
    main()
