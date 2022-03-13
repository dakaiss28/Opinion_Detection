""" this module gets tweets from twitter API, store them in CSV and then in database
after manual labelling"""
import json
import pyodbc
import tweepy
import pandas as pd

brands = ["covid"]


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


def get_tweets(api, file_name):
    df = pd.DataFrame(
        [],
        columns=[
            "tweet_id",
            "created_at",
            "content",
            "nb_retweets",
            "nb_fav",
            "topic",
            "target_label",
            "kmeans_res",
            "svm_res",
        ],
    )
    for brand in brands:
        for tweet in tweepy.Cursor(
            api.search_tweets, lang="en", q=brand, result_type="popular"
        ).items(6000):
            df = df.append(
                {
                    "tweet_id": tweet.id_str,
                    "created_at": tweet.created_at,
                    "content": tweet.text,
                    "nb_retweets": int(tweet.retweet_count),
                    "nb_fav": int(tweet.favorite_count),
                    "topic": brand,
                    "target_label": -1,
                    "kmeans_res": -1,
                    "svm_res": -1,
                },
                ignore_index=True,
            )
    df.to_csv(file_name)


def store_tweets(file_name, db_connexion):
    """store tweets in database"""
    data = pd.read_csv(file_name)
    df = pd.DataFrame(data)
    cursor = db_connexion.cursor()
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT INTO dbo.tweets values(?,?,?,?,?,?,?,?,?)",
            row["tweet_id"],
            row["created_at"],
            row["content"],
            row["nb_retweets"],
            row["nb_fav"],
            row["topic"],
            row["target_label"],
            row["kmeans_res"],
            row["svm_res"],
        )
        db_connexion.commit()
    db_connexion.close()


def retrieve_tweets(db_connexion):
    """get tweets from database"""
    query = "SELECT * FROM dbo.tweets"
    tweets_df = pd.read_sql(query, db_connexion)
    tweets_df.drop_duplicates()
    db_connexion.close()
    return tweets_df


def store_results(result, connexion, model):
    """store kmeans results in dataBase table"""
    cursor = connexion.cursor()
    for _, row in result.iterrows():
        cursor.execute(
            "UPDATE dbo.tweets SET " + model + " = ? WHERE tweet_id = ?",
            row["label"],
            row["id"],
        )
    connexion.commit()
    connexion.close()


def main():
    file_name = "tweets.csv"
    api, db_connexion = set_up("../twitter_token.json.txt")
    get_tweets(api, file_name)
    # store_tweets(file_name, db_connexion)


if __name__ == "__main__":
    main()
