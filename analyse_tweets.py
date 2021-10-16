import pyodbc 
import pandas as pd
import numpy as np
from get_tweets import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import collections

sns.set(style="darkgrid")
sns.set(font_scale=1.3)

def retrieve_tweets():
    db_connexion = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                        'Server=localhost;'
                        'Database=tweets;'
                        'Trusted_Connection=yes;')

    query = "SELECT * FROM dbo.tweets"
    tweets_df = pd.read_sql(query,db_connexion)
    tweets_df.drop_duplicates()
    db_connexion.close()
    return tweets_df

def simple_data_analysis(tweets_df):
    sns.catplot(x="label", data=tweets_df, kind="count", height=6, aspect=1.5, palette="PuBuGn_d").set(title = "label distribution for the whole dataset")
    brands = tweets_df['brand'].unique()
    for brand in brands:
        sns.catplot(x="label", data=tweets_df[(tweets_df.brand == brand)], kind="count", height=6, aspect=1.5, palette="PuBuGn_d").set(title = "label distribution for brand "+ brand)
    plt.show()

"""

def word_frequency(tweets_df):
    sr_clean = clean_df(tweets_df)
    sr_clean.sample(5)
    cv = CountVectorizer()
    bow = cv.fit_transform(sr_clean)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
    plt.show()
"""

def main():
    tweets_df = retrieve_tweets()
    tweets_df = clean_df(tweets_df)
    simple_data_analysis(tweets_df)
    #word_frequency(tweets_df)

if __name__ == "__main__":
    main()
