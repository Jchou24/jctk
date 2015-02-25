from nltk.corpus import treebank
from nltk.corpus import treebank_chunk
from nltk.corpus import nps_chat
from nltk.corpus import conll2000

corpus_description = {
    'conll2000':"""
        270k words of Wall Street Journal text, tagged and chunked.
        Wall Street Journal: an American English-language international daily newspaper with a special emphasis on business and economic news.
        CoNLL stands for the Conference on Computational Natural Language Learning. For the year 2000 conference, a shared task was undertaken to produce a corpus of chunks based on the Wall Street Journal corpus.
        ref: http://www.cnts.ua.ac.be/conll2000/chunking/
    """,
    'nps_chat':"""
        maintain by NUS(National University of Singapore)
        10k IM chat posts, POS-tagged and dialogue-act tagged
        Internet Chatroom Conversations corpus
        more data: http://wing.comp.nus.edu.sg:8080/SMSCorpus/history.jsp
    """,
    'treebank':"""
        The Penn Treebank Project: http://www.cis.upenn.edu/~treebank/home.html
        40k words of Wall Street Journal text, tagged and parsed
    """,
}

corpus_pickle = {
    'conll2000': {'train_name':'conll2000','train_sents':conll2000.tagged_sents('train.txt')},
    'conll2000_chunk': {'train_name':'conll2000_chunk','train_sents':conll2000.chunked_sents('train.txt')},
    'nps_chat': {'train_name':'nps_chat',"train_sents":nps_chat.tagged_posts()},
    'treebank': {'train_name':'treebank','train_sents':treebank.tagged_sents()},
    'treebank_chunk': {'train_name':'treebank_chunk','train_sents':treebank_chunk.chunked_sents()},
}

corpus = sorted( corpus_pickle.keys() )
