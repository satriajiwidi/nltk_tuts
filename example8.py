import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes/len(votes)
		return conf


documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

# random.shuffle(documents)

all_words = []
for word in movie_reviews.words():
	all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]
# print(word_features)
word_features = all_words.most_common(3000).keys()
# print(word_features)
# quit()

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets = [(find_features(rev), category) for rev, category in documents]

# positive
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# negative
training_set = featuresets[100:]
testing_set = featuresets[:100]

clf = nltk.NaiveBayesClassifier.train(training_set)
print('NB original Accuracy:', nltk.classify.accuracy(clf, testing_set) * 100)
# clf.show_most_informative_features(15)

MultinomialNB_clf = SklearnClassifier(MultinomialNB())
MultinomialNB_clf.train(training_set)
print('MultinomialNB_clf Accuracy:', nltk.classify.accuracy(MultinomialNB_clf, testing_set) * 100)

BernoulliNB_clf = SklearnClassifier(BernoulliNB())
BernoulliNB_clf.train(training_set)
print('BernoulliNB_clf Accuracy:', nltk.classify.accuracy(BernoulliNB_clf, testing_set) * 100)

LogisticRegression_clf = SklearnClassifier(LogisticRegression())
LogisticRegression_clf.train(training_set)
print('LogisticRegression_clf Accuracy:', nltk.classify.accuracy(LogisticRegression_clf, testing_set) * 100)

SGDClassifier_clf = SklearnClassifier(SGDClassifier())
SGDClassifier_clf.train(training_set)
print('SGDClassifier_clf Accuracy:', nltk.classify.accuracy(SGDClassifier_clf, testing_set) * 100)

# SVC_clf = SklearnClassifier(SVC())
# SVC_clf.train(training_set)
# print('SVC_clf Accuracy:', nltk.classify.accuracy(SVC_clf, testing_set) * 100)

LinearSVC_clf = SklearnClassifier(LinearSVC())
LinearSVC_clf.train(training_set)
print('LinearSVC_clf Accuracy:', nltk.classify.accuracy(LinearSVC_clf, testing_set) * 100)

NuSVC_clf = SklearnClassifier(NuSVC())
NuSVC_clf.train(training_set)
print('NuSVC_clf Accuracy:', nltk.classify.accuracy(NuSVC_clf, testing_set) * 100)

# with open('clf.pickle', 'wb') as clf_pickle:
# 	pickle.dump([clf, MultinomialNB_clf, BernoulliNB_clf, LogisticRegression_clf,
# 				SGDClassifier_clf, SVC_clf, LinearSVC_clf, NuSVC_clf],
# 				clf_pickle)
# 	print('clf created!')

# with open('training.pickle', 'wb') as training_set_pickle:
# 	pickle.dump(training_set, training_set_pickle)
# 	print('training set created!')



voted_classifier = VoteClassifier(clf, MultinomialNB_clf,
				   BernoulliNB_clf, LogisticRegression_clf,
				   SGDClassifier_clf, LinearSVC_clf, NuSVC_clf)
print('Voted Accuracy:', nltk.classify.accuracy(voted_classifier, testing_set) * 100)

print('Classification:', voted_classifier.classify(testing_set[0][0])
	  , '\tConfidence percent:', voted_classifier.confidence(testing_set[0][0]))