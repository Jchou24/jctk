from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(word):
    return lemmatizer.lemmatize(word)

def pos_lemmatize(word,pos):
    if pos == '' or pos == None:
        return lemmatizer(word)
    else:
        return lemmatizer.lemmatize(word, pos=pos)

if __name__ == '__main__':
    print lemmatize('cooking')
    print pos_lemmatize('cooking', pos='v')
