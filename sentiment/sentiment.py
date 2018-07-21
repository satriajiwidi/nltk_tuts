
# coding: utf-8

# In[64]:


import re
from string import punctuation
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from stopwords import get_stopwords
import corpus
from collections import Counter
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
import numpy as np


# In[65]:


# data loading
pos_tweets = corpus.pos
neg_tweets = corpus.neg
print(len(neg_tweets), len(pos_tweets))

# gabung semua tweets (pos dan neg)
tweets = [] # teks dan label
X = [] # teks saja
y = [] # label/sentiment saja
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))
    X.append(words_filtered)
    y.append(sentiment)
    
# print(X[1])
index_pos = [(index) for index, value in enumerate(y) if value == 'positive']
index_neg = [(index) for index, value in enumerate(y) if value == 'negative']

pos_tweets = [X[index] for index in index_pos]
pos_tweets_y = [y[index] for index in index_pos]
neg_tweets = [X[index] for index in index_neg]
neg_tweets_y = [y[index] for index in index_neg]


# In[76]:


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
    stopwords = get_stopwords()
#     all_words = [x for x in words]
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

# print(remove_stopwords(stem_words('Pemandangannya sungguh indah, saya sangat suka'.split())))


# In[77]:


all_words = sorted(set(remove_stopwords(stem_words(get_words(tweets)))))

word_features = get_word_features(all_words)
# print(nltk.FreqDist(all_words).most_common(4))
print(word_features)


# In[78]:


# mendapatkan fitur kata dari tiap dokumen
# menjadikan array one-hot
# dictionary {word: boolean presence}
def extract_features(document):
    document_words = set(document)
    document_words = remove_stopwords(stem_words(document))
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features


# In[80]:


# loading training data sets
# training_set = [(extract_features(rev), category) for rev, category in tweets]
training_set = [(extract_features(rev), category) for rev, category in tweets]
print(training_set[1])


# In[94]:


# khusus untuk penanganan imbalanced data
# 
# 

featuresets = [extract_features(tweet) for tweet in X]
featuresets_baru = []
temp = []
for index, i in enumerate(featuresets):
    for j, k in i.items():
        temp.append(1 if k == True else 0)
    featuresets_baru.append(temp)
    temp = []

from imblearn.over_sampling import SMOTE, ADASYN
y_baru = [1 if i == 'positive' else 0 for i in y]
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(featuresets_baru, y_baru)

# coba mengembalikan
for index, x in enumerate(X_resampled):
    XXX = [index for index, value in enumerate(x) if value == 1]
#     print(XXX)
    YYY = np.array([i for i in featuresets[0]])
    print(index, np.take(YYY, XXX), y_resampled[index])
#     break
# print(len(X_resampled))
# print(len(featuresets_baru))
# print(len(neg_tweets))


# In[71]:


# training data
# dengan menggunakan algoritma naive bayes classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# print(classifier.show_most_informative_features())

# validasi dengan menggunakan kalimat sendiri
tweet = 'ini sangat bahagia'
tweet = extract_features(tweet.split())

print(classifier.classify(tweet))

LinearSVC_clf = SklearnClassifier(LinearSVC())
LinearSVC_clf.train(training_set)
# print('LinearSVC_clf Accuracy:', nltk.classify.accuracy(LinearSVC_clf, training_set[:5]) * 100)
LinearSVC_clf.classify(tweet)

