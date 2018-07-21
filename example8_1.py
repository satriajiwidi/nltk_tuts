import pickle
import nltk

classifiers = []

with open('clf.pickle', 'rb') as clf_f:
	classifiers = [clf for clf in pickle.load(clf_f)]

with open('training.pickle', 'rb') as train:
	training_set = pickle.load(train)

# for clf in classifiers[:-1]:
	# print(clf)
