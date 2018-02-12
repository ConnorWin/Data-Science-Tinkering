import tweepy
from textblob import TextBlob
import csv

#Authenticate through Twitter Application Mangement
consumer_key= ''
consumer_secret= ''

access_token=''
access_token_secret=''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Retrieve Tweets
public_tweets = api.search('Trump')

#open csv
with open('results.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

    #analyze incoming tweet and write results with label based on how positive and opinionated it is
    for tweet in public_tweets:
        result = tweet.text.encode('utf-8').strip()
        analysis = TextBlob(tweet.text)
        if analysis.sentiment.polarity < 0.0 and analysis.sentiment.subjectivity > 0.2:
            label = "Bad: "
        else:
            label = "Good: "
        spamwriter.writerow([label + tweet.text.encode('utf-8').strip(), analysis.sentiment])