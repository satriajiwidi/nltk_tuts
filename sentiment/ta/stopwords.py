def get_stopwords():
    with open('stopwords.txt', 'r') as file:
        stopwords = file.read().split('\n')

    return stopwords