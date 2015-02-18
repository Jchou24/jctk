import dill
# import cPickle as pickle
import pickle
import nltk
from nltk.tag import hmm
from nltk.corpus import treebank
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger
from nltk.tag.sequential import ClassifierBasedPOSTagger

def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff

def hmm_tagger(train_sents):
    symbols = list( set([ word for sentence in train_sents for word,tag in sentence ]) )
    tag_set = list( set([ tag for sentence in train_sents for word,tag in sentence ]) )
    labelled_sequences = train_sents
    trainer = nltk.HiddenMarkovModelTrainer(tag_set, symbols)
    hmm = trainer.train_supervised(labelled_sequences[3000:],estimator=lambda fd, bins: nltk.LidstoneProbDist(fd, 0.1, bins))
    return hmm

class pos_tagger():
    def __init__(self,train_name='treebank',train_sents=treebank.tagged_sents(),using_tagger = "backoff", using_pickle=True):
        self.train_sents = train_sents
        self.valid_tagger = ["backoff","bayes","hmm"]
        # if using_tagger != "backoff" and using_tagger != "bayes":
        if using_tagger not in self.valid_tagger:
            raise Exception("valid taggers are:"+self.valid_tagger)
        self.using_tagger = using_tagger
        self.train_name = train_name
        if using_pickle:
            self.tagger = self._pickle_tagger()
        else:
            self.tagger = self._tagger_dict(self.using_tagger,self.train_sents)

    def __get__(self):
        return self.tagger
        # return "hi"

    def _tagger_dict(self,using_tagger,train_sents):
        tagger_dict = {
            "backoff": backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=DefaultTagger('NN')),
            "bayes":   ClassifierBasedPOSTagger(train=train_sents),
            "hmm":     hmm_tagger(train_sents),

        }
        return tagger_dict[using_tagger]

    def _pickle_tagger(self):
        import os.path
        pickle_name = os.path.join( os.path.dirname(__file__), "pos_tagger_pickle", self.using_tagger+'.'+self.train_name+'.pickle' )

        # pickle_name = 'pos_tagger_pickle/'+self.using_tagger+'.'+self.train_name+'.pickle'
        try:
            f = open(pickle_name, 'r')
            tagger = pickle.load(f)
        except:
            tagger = self._tagger_dict(self.using_tagger,self.train_sents)
            try:
                f = open(pickle_name, 'w')
                pickle.dump(tagger, f)
                f.close()
            except:
                pass
        return tagger

    def _tag_list(self,tagged_sents):
        return [tag for sent in tagged_sents for (word, tag) in sent]

    def _apply_tagger(self,tagger, corpus):
        return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]

    def cross_calidation(self,nfold=3):
        fold_size = len(self.train_sents)/nfold
        for i in range(nfold):
            train_sents = self.train_sents[:i*fold_size] + self.train_sents[(i+1)*fold_size:]
            test_sents = self.train_sents[i*fold_size:][:fold_size]
            # print train_sents
            # print test_sents

            # if self.using_tagger == "backoff":
            #     tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=DefaultTagger('NN'))
            # elif self.using_tagger == "bayes":
            #     tagger = ClassifierBasedPOSTagger(train=train_sents)
            tagger = self._tagger_dict(self.using_tagger,train_sents)

            print "="*30 + " Evaluate Fold: "+str(i)+" " + "="*30
            ans_tag = self._tag_list(test_sents)
            test_tag = self._tag_list(self._apply_tagger(tagger,test_sents))
            self._score(ans_tag,test_tag)

    def _score(self,ans_tag,test_tag):
        ans_tag_set = set(ans_tag)
        test_tag_set = set(test_tag)
        print 'using_tagger:', self.using_tagger
        print ' ori tag set:', ans_tag_set
        print 'test tag set:', test_tag_set
        print '    Accuracy:', nltk.accuracy(ans_tag, test_tag)
        print '   Precision:', nltk.precision(ans_tag_set, test_tag_set)
        print '      Recall:', nltk.recall(ans_tag_set, test_tag_set)
        print '   F-Measure:', nltk.f_measure(ans_tag_set, test_tag_set)

        # ConfusionMatrix input format:
            #     Ans(arg0): NN VB NN NN
            #     Tag(arg1): NN VB VB VB
        cm = nltk.ConfusionMatrix(ans_tag,test_tag)
        print cm.pp(sort_by_count=True, show_percents=True, truncate=9)
        print "="*78

    def evaluation(self):
        ans_tag = self._tag_list(self.train_sents)
        test_tag = self._tag_list(self._apply_tagger(self.tagger,self.train_sents))
        self._score(ans_tag,test_tag)

if __name__ == '__main__':
    # postagger = pos_tagger(using_tagger='bayes')# 0.977651078708
    #0.91
    postagger = pos_tagger() # 0.992361635345
    # 0.87
    # postagger = pos_tagger(using_tagger='hmm') # 0.85510946005
    # 0.008
    # postagger.cross_calidation(3)
    # postagger.evaluation()

    # demo_pos()

    # from nltk.corpus import brown
    # print brown.tagged_sents(categories='news')[:10]
    # print treebank.tagged_sents()[:10]
    # print load_pos(10)[0]
