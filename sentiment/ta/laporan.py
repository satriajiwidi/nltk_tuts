import json
with open('data.json', 'r') as file:
	raw_data = json.load(file)

raw_data = raw_data[:150]
# raw_data = [
# 	{"class": 1, "text": "Makanannya enak, kamar bersih, harga terjangkau"},
# 	{"class": 1, "text": "Pelayanannya bagus, makanannya enak"},
# 	{"class": 1, "text": "Kamar luas, menu makanan lengkap."},
# 	{"class": 1, "text": "Pelayanan yang ramah dan hotel bersih"},
# 	{"class": 0, "text": "Kamarnya kotor sekali."},
# 	{"class": 0, "text": "Harga terlalu mahal"},
# 	{"class": 0, "text": "Kamar pengap bau dan kotor"},
# ]
# raw_data = [
# 	{"class": 1, "text": "Pelayanannya bagus, makanannya enak, nice"},
# 	{"class": 1, "text": "Pelayanan yang ramah dan keadaaan hotel yang bersih"},
# 	{"class": 1, "text": "Hotel murah dengan kualitas bagus lokasi dekat dengan pusat kota dan tempat makan"},
# 	{"class": 0, "text": "Banyak jentik nyamuk, sempat pindah kamar, ternyata setelah pindah kamar, sama saja banyak jentik nyamuk"},
# 	{"class": 0, "text": "Hari pertama Makanan kehabisan, AC tidak dingin beberapa karyawan tidak ada senyum salam sapa terutama security"},
# ]
# print(len([minor['class'] for minor in raw_data if minor['class'] == 0]))
import numpy as np
Y = [data['class'] for data in raw_data]

texts = [data['text'] for data in raw_data]
# indices = [1, 5, 4, 7, 8]
# texts = [texts[i] for i in indices]

from stopwords import get_stopwords
stopwords = get_stopwords()

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

texts_normalized = []

if __name__ == "__main__":
	import preprocess
	kamus = preprocess.get_kamus()
	import re
	for text in texts:
		text_normalized = []
		text_tokenized = []
		text_stopwords_removed = []
		text_stemmed = []
		text_normalisasi = []

		for word in text.split():
			word = word.lower()
			word = re.match('[a-z]+', word)

			if word is not None:
				word = word.group(0)
				text_tokenized.append(word)

				# word = preprocess.correction([word], kamus)[word]
				# text_normalisasi.append(word)

				if word not in stopwords:
					text_stopwords_removed.append(word)
					word = stemmer.stem(word)
					text_stemmed.append(word)
					text_normalized.append(word)
		
		texts_normalized.append(' '.join(text_normalized))
	# 	print('text\n', text)
	# 	print('text_tokenized\n', text_tokenized)
	# 	# print('text_normalisasi\n', text_normalisasi)
	# 	print('text_stopwords_removed\n', text_stopwords_removed)
	# 	print('text_stemmed\n', text_stemmed)
	# 	print()

	# print('texts_normalized', texts_normalized)
	# print()

	all_words = [word for sentence in texts_normalized
				 for word in sentence.split()]

	from nltk import FreqDist
	fd = FreqDist(all_words) # sebelum di-set, bentuk object freqdist

	all_words = list(sorted(set(all_words)))

	hapaxes = fd.hapaxes()

	# print('hapaxes', hapaxes)

	all_words = [word for word in all_words
				 if len(word) > 2
				 and word not in hapaxes]

	# print('features:')
	# print(len(all_words), all_words)

	from vectorizers import binary_vectorizer, count_vectorizer, tfidf_vectorizer
	# biner = binary_vectorizer(texts_normalized, all_words)
	# count = count_vectorizer(texts_normalized, all_words)
	# tfidf = tfidf_vectorizer(texts_normalized, all_words)

	# print(biner.tolist())
	# print(count.tolist())
	# print(tfidf.tolist())


	array_fitur = ['biner', 'count', 'tfidf']
	array_fitur = ['count', 'tfidf']

	vectorizers = dict(zip(array_fitur, [
		# binary_vectorizer,
		count_vectorizer,
		tfidf_vectorizer
	]))

	X = {fitur: vectorizers[fitur](texts_normalized, all_words) for fitur in array_fitur}

	x1 = list(sorted(set(X['count'].flatten())))
	x2 = list(sorted(set(X['tfidf'].flatten())))
	print(len(x1))
	print(len(x2))

	# from smote import SMOTE as smote
	# data_resampled = {fitur: smote(X[fitur], Y, 100, k=3, random_seed=10) for fitur in array_fitur}
	# X_resampled = {fitur: data_resampled[fitur][0] for fitur in array_fitur}
	# Y_resampled = data_resampled[array_fitur[0]][1]

	# # print(X, Y)
	# # print(X_resampled, Y_resampled)

	from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	# print(Y[1:])
	# clf.fit(X['count'][1:], Y[1:])
	# print(clf.predict_log_proba(X['count'][:1]))
	# print(clf.predict_proba(X['count'][:1]))
	# print(clf.predict(X['count'][:1]))
