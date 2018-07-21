import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

sample_text = "The book is read by Widi Satriaji at the room. And then read again by Bubu Baba."

# print(sample_text)

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for item in tokenized:
			words = nltk.word_tokenize(item)
			tagged = nltk.pos_tag(words)

			name_ent = nltk.ne_chunk(tagged, binary=True)

			print(name_ent)

			name_ent.draw()

	except Exception as e:
		print(str(e))

process_content()