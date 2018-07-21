import tweepy
from textblob import TextBlob

consumer_key = 'lhcLm41hWeUnHVqtfrji5xb9f'
consumer_secret = 'dGsLx527Xo8AjnZH5phqUlGjxqdXd8tKOumvKvfQthhDl8YwTX'

access_token = '567058088-11nke413Y7qwlq0SGGXEGlUc7Lg0poCDoOCPQ3pX'
access_token_secret = 'eRh0lHe7YuBrIby7d8PLMA5UeN8gtktX7IikYASpOWRAo'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Jokowi')
# public_tweets = api.home_timeline()

for tweet in public_tweets:
	print(tweet.text)
	# analysis = TextBlob(tweet.text)
	# print(analysis.sentiment)
	print('==========================================')

# user = api.get_user('satriajiwidi')

# print(user.screen_name)
# print(user.followers_count)
# for index, friend in enumerate(tweepy.Cursor(api.followers).items()):
# 	print(index, friend.screen_name)

# api.update_status('Second try. Tweeted using Tweepy python library for Twitter API.')

# for index, status in enumerate(tweepy.Cursor(api.user_timeline).items(30)):
# 	print(index+1, status.text)
