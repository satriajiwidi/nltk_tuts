import nltk
from nltk.corpus import wordnet

# syns = wordnet.synsets('healthy')

# # synset
# print(syns[0].name())

# # just a word
# print(syns[0].lemmas()[0].name())

# #  definition
# print(syns[0].definition())

# # examples
# print(syns[0].examples())

# sinonyms = []
# antonyms = []

# for syn in wordnet.synsets('boat'):
# 	for l in syn.lemmas():
# 		# print(l)
# 		sinonyms.append(l.name())
# 		if l.antonyms():
# 			antonyms.append(l.antonyms()[0].name())

# print(set(sinonyms))
# print(set(antonyms))

word1 = wordnet.synset('boat.n.01')
word2 = wordnet.synset('ship.n.01')

sentence = 'The world is not big enough to live in. We need to move to other planet soon!'

words = nltk.word_tokenize(sentence)
tagged_words = nltk.pos_tag(words)

# print(tagged_words)

# mengganti kata benda dan kata sifat yang memiliki sinonim
# for word, tipe in tagged_words:
# 	if tipe == 'NN' or tipe == 'JJ':
# 		syns = wordnet.synsets(word)
# 		if syns:
# 			word = syns[0].lemmas()[0].name()

# 	print(word, end=" ")

# print(word1, word2, word1.wup_similarity(word2))