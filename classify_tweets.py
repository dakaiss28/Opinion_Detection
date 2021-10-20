from get_tweets import *
from analyse_tweets import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

def classify_tweets(tweets_df):
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(tweets_df['content'],tweets_df['label'],test_size=0.3)
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(tweets_df['content'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    SVM.fit(Train_X_Tfidf,Train_Y) # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf) # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

def main():
    #fecth_tweets()
    tweets_df = retrieve_tweets()
    df = clean_df(tweets_df)
    #simple_data_analysis(df)
    #word_frequency(df)
    #classify_tweets(df)

if __name__ == "__main__":  
    main()
