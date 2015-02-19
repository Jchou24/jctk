from tokenizer import *
from nltk import Tree

def parse2chunks(list_of_string,tagger,chunker):
    sentences = []
    for para in list_of_string:
        sentences.extend( para2sents(para.lower()) )
    # print sentences

    chunk_sentence = [ sent2chunks(sentence) for sentence in sentences ]
    chunk_sentence = [ sent2words(c) for cs in chunk_sentence for c in cs ]
    # print chunk_sentence

    tag_sentence = [ tagger.tag(cs) for cs in chunk_sentence ]
    # print tag_sentence

    tag_chunk_sentence = [ chunker.parse(ts) for ts in tag_sentence ]
    # print tag_chunk_sentence
    # for tcs in tag_chunk_sentence:
    #     tcs.draw()

    return Tree('paragraph',tag_chunk_sentence)
