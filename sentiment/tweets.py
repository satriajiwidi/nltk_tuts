import tweepy

consumer_key = 'lhcLm41hWeUnHVqtfrji5xb9f'
consumer_secret = 'dGsLx527Xo8AjnZH5phqUlGjxqdXd8tKOumvKvfQthhDl8YwTX'

access_token = '567058088-11nke413Y7qwlq0SGGXEGlUc7Lg0poCDoOCPQ3pX'
access_token_secret = 'eRh0lHe7YuBrIby7d8PLMA5UeN8gtktX7IikYASpOWRAo'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

file = open('tweets.txt', 'a')

while True:
	public_tweets = api.search('Jokowi')
	for tweet in public_tweets:
		for line in tweet.text.splitlines():
			print(line)
			file.write(line + ' ')
		file.write('\n')

file.close()
