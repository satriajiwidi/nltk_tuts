import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

with open('Indonesian_Manually_Tagged_Corpus.tsv') as file:
	data = []
	for line in csv.reader(file, dialect="excel-tab"):
		if len(line) == 2:
			data.append(line)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

data = [stemmer.stem(d[0]) for d in data
		if d[1] == 'NEG' or
		d[1] == 'CC' or
		d[1] == 'FW' or
		d[1] == 'JJ']
words = list(set(data))

import pickle
with open('pos_tag_indo.pkl', 'wb') as file:
	pickle.dump(words, file)