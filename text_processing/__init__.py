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

from normalizer import *
from tokenizer import *
from stemmer import *
from lemmatizer import *
from pos_tagger import pos_tagger
from chunker import TagChunker
from corpus import *
