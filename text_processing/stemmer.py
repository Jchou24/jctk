from nltk.stem import PorterStemmer

def stem(word):
    return PorterStemmer.stem(word)
