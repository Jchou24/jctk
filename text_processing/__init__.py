# Modules/Methods:
#     normalizer
#     tokenizer
#         paragraph to sentence: para2sents
#         sentence to chunk:     sent2chunks
#         sentence to word:      sent2words
#     stemmer:    stem
#     lemmatizer: lemmatize
#     pos tagger: pos_tagger
#     chunker:    TagChunker
#     corpus

import codecs
import os.path

from normalizer import *
from tokenizer import *
from stemmer import *
from lemmatizer import *
from pos_tagger import pos_tagger
from chunker import TagChunker
from corpus import *

en_stop_words = []

en_stop_words_file = os.path.join( os.path.dirname(__file__), "english.stop" )
with codecs.open(en_stop_words_file,'r') as op:
	en_stop_words = [ line.strip() for line in op ]