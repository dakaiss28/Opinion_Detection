"""This module classify the tweets"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from src.get_tweets import clean_df
from src.analyse_tweets import retrieve_tweets, word_frequency


def classify_tweets(tweets_df):
    """This method convert the twwets text into vectors and
    then classify them as neutral,positive or negative using SVM classifier"""
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
        tweets_df["content"], tweets_df["label"], test_size=0.3
    )
    svm_classifier = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")

    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(tweets_df["content"])
    train_x_tfidf = tfidf_vect.transform(train_x)
    test_x_tfidf = tfidf_vect.transform(test_x)

    # predict the labels on validation dataset
    svm_classifier.fit(train_x_tfidf, train_y)

    # Use accuracy_score function to get the accuracy
    predictions_svm = svm_classifier.predict(test_x_tfidf)
    print(f"SVM Accuracy -> {accuracy_score(predictions_svm, test_y)*100}")


def main():
    """main method"""
    # fecth_tweets()
    tweets_df = retrieve_tweets()
    data_frame = clean_df(tweets_df)
    # simple_data_analysis(data_frame)
    word_frequency(data_frame)
    # classify_tweets(data_frame)


if __name__ == "__main__":
    main()
