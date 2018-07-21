from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']

# for word in example_words:
# 	print(ps.stem(word))

# neW_text = 'It is very important to be pythonly while you are pythoning with python. ALl pythoners have python poorly at least once.'
neW_text = "Riding reading taking"

words = word_tokenize(neW_text)

for word in words:
	print(ps.stem(word))