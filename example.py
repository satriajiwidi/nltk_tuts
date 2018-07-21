from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_text = 'this is an example of bunch of words. this is gonna be tokenized.'

stop_words = set(stopwords.words('english'))

words = word_tokenize(example_text)

filtered_sentence = []

# for word in words:
# 	if word not in stop_words:
# 		filtered_sentence.append(word)

filtered_sentence = [word for word in words if word not in stop_words]

print(filtered_sentence)