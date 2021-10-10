import tweepy
import json
import pyodbc 
import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
dico = {}

def set_up(file):
    with open(file) as f:
        tokens = json.load(f)

        consumer_key = tokens["Key"]
        consumer_secret = tokens["Secret"]

        auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

        api = tweepy.API(auth)
    
    db_connexion = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                        'Server=localhost;'
                        'Database=tweets;'
                        'Trusted_Connection=yes;')
    
    return (api,db_connexion)
    """
    cursor = db_connexion.cursor()
    cursor.execute('SELECT * FROM table_name')

    for i in cursor:
        print(i)
    """

def clean_text(text):
    text.lower()
    text = re.sub('RT','',text)
    text = re.sub('@', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)
    
    
    #Tokenize the data
    text = nltk.word_tokenize(text)
    #Remove stopwords
    text = [word for word in text if word not in stop_words]
    return text

def lemmatize_text(text):
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [lemmatizer.lemmatize(t, 'v') for t in text]
    return text

def word_int(word):
    if word not in dico:
        id = len(dico) + 1
        dico[word] = id
    return dico[word]


def label_text(text):
    analyser = SentimentIntensityAnalyzer()
    res = analyser.polarity_scores(text)
    neg = res['neg']
    pos = res['pos']
    neu = res['neu']
    if (neg > pos and neg > neu):
        return -1
    elif(pos >neg and pos > neu):
        return 1
    else:
        return 0
    
def text_to_vector(text):
    return list(map(word_int,text))

def main():
    (api,db_connexion) = set_up("twitter_token.json.txt")
    for tweet in tweepy.Cursor(api.search_tweets,lang = "en", q='apple').items(2):
        print(tweet.text)
        label_text(tweet.text)
        clean = clean_text(tweet.text)
        text = lemmatize_text(clean)
        print(label_text(tweet.text))
        print(text_to_vector(text))
        """
        print(tweet.created_at)
        print(tweet.id)
        print(tweet.text)
        print(tweet.entities["hashtags"])
        print(tweet.user.time_zone)
        print(tweet.retweet_count)
        print(tweet.favorite_count)

    """
        

if __name__ == "__main__":
    main()
