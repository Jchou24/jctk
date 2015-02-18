# Modules/Methods:
#     tokenizer
#         paragraph to sentence: para2sents
#         sentence to chunk:     sent2chunks
#         sentence to word:      sent2words

from nltk.tokenize import sent_tokenize
from nltk.tokenize import regexp_tokenize
import nltk.data

def para2sents(paragraph,language="english"):
    try:
        pickle_path = 'tokenizers/punkt/'+language+'.pickle'
        tokenizer = nltk.data.load(pickle_path)
        return tokenizer.tokenize(paragraph)
    except:
        return sent_tokenize(paragraph)

def sent2chunks(sentence):
    return [ s.strip() for s in sentence.split(",") ]

def sent2words(sentence,cond="[\w']+"):
    # regexp_tokenize("Can't is a contraction.", "[\w']+")
    # ["Can't", 'is', 'a', 'contraction']
    return regexp_tokenize(sentence, cond)

if __name__ == '__main__':
    paragraph = "Hi, how are you. Are you ok?"
    # paragraph = "Drizzle the rest oil in the pan, toss in the chopped tomatoes, stir and add some salt. Let it cook for about 2 minutes, or till the tomatoes start to reduce its size for a little bit and the juices come out."
    sentences = para2sents(paragraph)
    print sentences
    words = [ sent2words(sentence) for sentence in sentences]
    print words
    chunks = [ sent2chunks(sentence) for sentence in sentences]
    print chunks

