import tweepy
import json
import pyodbc 

with open("twitter_token.json.txt") as f:
   tokens = json.load(f)

consumer_key = tokens["Key"]
consumer_secret = tokens["Secret"]

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth)

db_connexion = pyodbc.connect('Driver={SQL Server};'
                      'Server=localhost;'
                      'Database=tweets;'
                      'Trusted_Connection=yes;')

"""
cursor = db_connexion.cursor()
cursor.execute('SELECT * FROM table_name')

for i in cursor:
    print(i)
"""
for tweet in tweepy.Cursor(api.search_tweets,lang = "en", q='apple').items(1):
    print(tweet.created_at)
    print(tweet.id)
    print(tweet.text)
    print(tweet.entities["hashtags"])
    print(tweet.user.time_zone)
    print(tweet.retweet_count)
    print(tweet.favorite_count)
