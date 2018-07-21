from string import punctuation
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from stopwords import get_stopwords
import corpus

# data loading
pos_tweets = corpus.pos
neg_tweets = corpus.neg

# gabung semua tweets (pos dan neg)
tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))

# tokenisasi
# mentransformasi seluruh kalimat dalam corpus menjadi array dari
# kata-kata
def get_words(tweets):
    all_words = []
    for (words, sentiment) in tweets:
    	all_words.extend(words)
    return all_words

# mendapatkan word features
# mengurutkan kata-kata dari frekuensi kemunculan tertinggi
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

# hapus stopwords dan tanda baca
def remove_stopwords(words):
	translator = str.maketrans('','', punctuation)
	stopwords = get_stopwords()
	all_words = [word.translate(translator) for word in words
				 if word not in stopwords]
	return all_words

# stemming
# mengubah kata-kata menjadi kata dasarnya
# menghilangkan imbuhan pada kata
def stem_words(words):
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	all_words = [stemmer.stem(word) for word in words]
	return all_words

all_words = stem_words(remove_stopwords(get_words(tweets)))

word_features = get_word_features(all_words)

# mendapatkan fitur kata dari tiap dokumen
# menjadikan array one-hot
# dictionary {word: boolean presence}
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features


# loading training data sets
training_set = nltk.classify.apply_features(extract_features, tweets)

# training data
# dengan menggunakan algoritma naive bayes classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

print(classifier.show_most_informative_features())


# validasi dengan menggunakan kalimat sendiri
tweet = 'Sungguh indah'
print(classifier.classify(extract_features(tweet.split())))
