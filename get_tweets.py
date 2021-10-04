import tweepy

print("Enter consumer Key")
consumer_key = input()
print("Enter consumer Secret")
consumer_secret = input()

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth)
for tweet in tweepy.Cursor(api.search_tweets,lang = "en", q='facebook').items(1):
    print(tweet.text)