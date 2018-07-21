from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw('bible-kjv.txt')

tokens = sent_tokenize(sample)

for sent in tokens[:20]:
	print('#', sent)