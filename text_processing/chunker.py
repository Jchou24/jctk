import nltk.chunk, itertools
from nltk.corpus import treebank_chunk, conll2000
from pos_tagger import pos_tagger, backoff_tagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger



def conll_tag_chunks(chunk_sents):
    tagged_sents = [nltk.chunk.tree2conlltags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

class TagChunker(nltk.chunk.ChunkParserI):
    def __init__(self, train_name='treebank_chunk',train_sents=treebank_chunk.chunked_sents(),using_tagger = "backoff"):
        #  train_name: treebank_chunk,                 conll2000_chunk
        # train_sents: treebank_chunk.chunked_sents(), conll2000.chunked_sents('train.txt')
        self.train_sents = train_sents
        self.train_chunks = conll_tag_chunks(train_sents)
        self.train_name = train_name
        self.using_tagger = using_tagger
        self._pos_tagger = pos_tagger(train_name=train_name,train_sents=self.train_chunks,using_tagger = using_tagger )
        self.tagger = self._pos_tagger.tagger

    def parse(self, tagged_sent):
        if not tagged_sent:
            return None
        (words, tags) = zip(*tagged_sent)
        chunks = self.tagger.tag(tags)
        wtc = itertools.izip(words, chunks)
        return nltk.chunk.conlltags2tree([(w,t,c) for (w,(t,c)) in wtc])

    def cross_validation(self,nfold=3):
        tmp_tagger = self.tagger
        fold_size = len(self.train_sents)/nfold
        for i in range(nfold):
            train_sents = self.train_sents[:i*fold_size] + self.train_sents[(i+1)*fold_size:]
            test_sents = self.train_sents[i*fold_size:][:fold_size]
            train_sents = conll_tag_chunks(train_sents)
            self.tagger = pos_tagger(train_sents=train_sents,using_tagger = self.using_tagger,using_pickle=False ).tagger
            score = self.evaluate(test_sents)
            print "="*30 + " Evaluate Fold: "+str(i)+" " + "="*30
            print ' Accuracy:', score.accuracy()
            print 'Precision:', score.precision()
            print '   Recall:', score.recall()
            print 'F-Measure:', score.f_measure()
        self.tagger = tmp_tagger

    def evaluation(self):
        score = self.evaluate(self.train_sents)
        print "="*78
        print ' Accuracy:', score.accuracy()
        print 'Precision:', score.precision()
        print '   Recall:', score.recall()
        print 'F-Measure:', score.f_measure()

if __name__ == '__main__':
    chunker = TagChunker(using_tagger = "backoff")
    chunker.evaluation()
    chunker = TagChunker(using_tagger = "bayes")
    chunker.evaluation()
    chunker = TagChunker(train_name='conll2000_chunk',train_sents=conll2000.chunked_sents('train.txt'),using_tagger = "backoff")
    chunker.evaluation()
    chunker = TagChunker(train_name='conll2000_chunk',train_sents=conll2000.chunked_sents('train.txt'),using_tagger = "bayes")
    chunker.evaluation()

    # chunker.cross_validation()
    # chunker.evaluation()

