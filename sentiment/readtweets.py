# import sentiment as sen
# from sentiment import stem_words, remove_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from stopwords import get_stopwords
import re

# hapus stopwords dan tanda baca
def remove_stopwords(words):
    stopwords = get_stopwords()
    all_words = [re.sub(r'[^\w\s]','', x) for x in words] # remove punctuation
    all_words = [x for x in all_words if x not in stopwords]
    return all_words

# stemming
# mengubah kata-kata menjadi kata dasarnya
# menghilangkan imbuhan pada kata
def stem_words(words):
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	all_words = [stemmer.stem(word) for word in words]
	return all_words

data = []
temp = []
for tweet in open('tweets.txt'):
	for line in tweet.splitlines():
		temp.append(line)
		# print(temp)
	data.append(temp)
	temp = []

words = []
for tweetlist in data:
	for tweet in tweetlist:
		# hilangkan username
		tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweet).split())
		# hilangkan angka
		tweet = ' '.join(re.sub("[0-9]+"," ", tweet).split())
		for word in tweet.split():
			words.append(word)

all_words = sorted(set(remove_stopwords(stem_words(words))))
for word in all_words:
	print(word)