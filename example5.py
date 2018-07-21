from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('feet'))
print(lemmatizer.lemmatize('tooth'))
print(lemmatizer.lemmatize('children'))
print(lemmatizer.lemmatize('carries'))
print(lemmatizer.lemmatize('booking', 'v'))