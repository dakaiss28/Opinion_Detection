""" module to make analysis on the tweets retrieved"""
import seaborn as sns
import pyodbc
import pandas as pd
import nltk


sns.set(style="darkgrid")
sns.set(font_scale=1.3)


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


def simple_data_analysis(tweets_df):
    """analyses the label distribution on data retrieved"""
    sns.catplot(
        x="label",
        data=tweets_df,
        kind="count",
        height=6,
        aspect=1.5,
        palette="PuBuGn_d",
    ).set(title="label distribution for dataset")
    brands = tweets_df["brand"].unique()
    # plt.savefig('labelDistribution.png')
    for brand in brands:
        sns.catplot(
            x="label",
            data=tweets_df[(tweets_df.brand == brand)],
            kind="count",
            height=6,
            aspect=1.5,
            palette="PuBuGn_d",
        ).set(title="label distribution for brand " + brand)
        # plt.show()
        # plt.savefig('labelDistribution'+brand+'.png')


def word_frequency(tweets_df):
    """analyses most frequend words in the dataset"""
    all_words = []
    for _, row in tweets_df.iterrows():
        all_words = all_words + row["content"].split()
    nlp_words = nltk.FreqDist(all_words)
    nlp_words.plot(20, color="salmon", title="Word Frequency")
