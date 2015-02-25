# import dill
import cPickle as pickle
# import pickle
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
            "backoff": backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=DefaultTagger("NN")),
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

    def cross_validation(self,nfold=3):
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
    # postagger.cross_validation(3)
    postagger.evaluation()

    # from nltk.corpus import brown
    # print brown.tagged_sents(categories='news')[:10]
    # print treebank.tagged_sents()[:10]
    # print load_pos(10)[0]

# Backoff
# ============================== Evaluate Fold: 0 ==============================
# using_tagger: backoff
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u'VBN', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'RP', u'$', u'NN', u'FW', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'CD', u'-NONE-', u'EX', u'IN', u'WP$', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'SYM', u'UH'])
# test tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u'VBN', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'RP', u'$', 'NN', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'CD', u'-NONE-', u'EX', u'IN', u'WP$', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR'])
#     Accuracy: 0.851935668809
#    Precision: 1.0
#       Recall: 0.933333333333
#    F-Measure: 0.965517241379
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |             N                    N             N               |
#        |      N      N      I      D      E      J      N               |
#        |      N      P      N      T      -      J      S      ,      . |
# -------+----------------------------------------------------------------+
#     NN | <11.9%>  0.0%   0.0%   0.0%      .   0.1%   0.0%      .      . |
#    NNP |   4.1%  <5.8%>  0.0%   0.0%      .   0.1%   0.0%      .      . |
#     IN |   0.1%      .  <9.6%>  0.0%      .   0.0%      .      .      . |
#     DT |   0.0%      .   0.0%  <8.1%>     .      .      .      .      . |
# -NONE- |   0.3%      .      .      .  <6.2%>     .      .      .      . |
#     JJ |   1.9%   0.1%   0.0%   0.0%      .  <3.8%>  0.0%      .      . |
#    NNS |   1.7%   0.0%      .      .      .   0.0%  <4.2%>     .      . |
#      , |   0.0%      .      .      .      .      .      .  <4.7%>     . |
#      . |      .      .      .      .      .      .      .      .  <3.9%>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# ============================== Evaluate Fold: 1 ==============================
# using_tagger: backoff
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'VB', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'FW', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'``', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'UH'])
# test tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', 'NN', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR'])
#     Accuracy: 0.866981594032
#    Precision: 1.0
#       Recall: 0.955555555556
#    F-Measure: 0.977272727273
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |                    N             N      N                      |
#        |      N      I      N      D      E      N      J               |
#        |      N      N      P      T      -      S      J      ,      . |
# -------+----------------------------------------------------------------+
#     NN | <12.3%>  0.0%   0.0%   0.0%      .   0.0%   0.1%      .      . |
#     IN |   0.0%  <9.3%>     .   0.0%      .      .   0.0%      .      . |
#    NNP |   2.9%      .  <6.2%>  0.0%      .   0.0%   0.1%      .      . |
#     DT |   0.0%   0.0%      .  <8.3%>     .      .   0.0%      .      . |
# -NONE- |   0.5%      .      .      .  <6.1%>     .      .      .      . |
#    NNS |   1.4%      .   0.0%      .      .  <4.6%>  0.0%      .      . |
#     JJ |   1.6%   0.0%   0.1%   0.0%      .      .  <3.9%>     .      . |
#      , |      .      .      .      .      .      .      .  <5.1%>     . |
#      . |      .      .      .      .      .      .      .      .  <3.7%>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# ============================== Evaluate Fold: 2 ==============================
# using_tagger: backoff
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u',', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'FW', u'POS', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR'])
# test tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u',', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', 'NN', u'POS', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'UH'])
#     Accuracy: 0.870955938405
#    Precision: 0.976744186047
#       Recall: 0.976744186047
#    F-Measure: 0.976744186047
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |                    N             N      N                      |
#        |      N      I      N      D      E      N      J      C        |
#        |      N      N      P      T      -      S      J      D      , |
# -------+----------------------------------------------------------------+
#     NN | <13.3%>     .   0.0%   0.0%      .   0.1%   0.2%      .      . |
#     IN |   0.0%  <9.4%>     .   0.0%      .      .   0.0%      .      . |
#    NNP |   2.9%      .  <5.6%>  0.0%      .   0.0%   0.1%      .      . |
#     DT |   0.0%   0.0%      .  <7.7%>     .      .      .      .      . |
# -NONE- |   0.0%      .      .      .  <6.6%>     .      .      .      . |
#    NNS |   1.2%      .   0.0%      .      .  <4.6%>     .      .      . |
#     JJ |   1.4%   0.0%   0.1%      .      .      .  <4.1%>     .      . |
#     CD |   1.5%      .      .      .      .   0.0%      .  <3.5%>     . |
#      , |      .      .      .      .      .      .      .      .  <4.8%>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# Bayes
# ============================== Evaluate Fold: 0 ==============================
# using_tagger: bayes
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u'VBN', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'RP', u'$', u'NN', u'FW', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'CD', u'-NONE-', u'EX', u'IN', u'WP$', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'SYM', u'UH'])
# test tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u'VBN', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'FW', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'PDT', u'RBS', u'RBR', u'CD', u'-NONE-', u'EX', u'IN', u'WP$', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'UH'])
#     Accuracy: 0.90869315953
#    Precision: 0.977272727273
#       Recall: 0.955555555556
#    F-Measure: 0.966292134831
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |             N                    N             N               |
#        |      N      N      I      D      E      J      N               |
#        |      N      P      N      T      -      J      S      ,      . |
# -------+----------------------------------------------------------------+
#     NN |  <9.9%>  0.1%   0.0%   0.0%      .   0.8%   0.1%      .      . |
#    NNP |   0.4%  <7.9%>  0.0%   0.1%   0.0%   0.5%   0.2%      .      . |
#     IN |   0.0%   0.0%  <9.7%>  0.0%      .   0.0%      .      .      . |
#     DT |   0.0%      .   0.0%  <8.0%>     .   0.0%      .      .      . |
# -NONE- |      .      .      .      .  <6.5%>     .      .      .      . |
#     JJ |   0.3%   0.3%   0.0%   0.0%   0.0%  <4.6%>  0.0%      .      . |
#    NNS |   0.0%   0.0%      .      .      .   0.0%  <5.8%>     .      . |
#      , |      .      .      .   0.0%      .      .      .  <4.7%>     . |
#      . |      .      .      .      .      .      .      .      .  <3.9%>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# ============================== Evaluate Fold: 1 ==============================
# using_tagger: bayes
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'VB', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'FW', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'``', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'UH'])
# test tag set: set([u'PRP$', u'VBG', u'VBD', u'VB', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'``', u'WRB', u'CC', u'PDT', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR'])
#     Accuracy: 0.923553302417
#    Precision: 1.0
#       Recall: 0.911111111111
#    F-Measure: 0.953488372093
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |                    N             N      N                      |
#        |      N      I      N      D      E      N      J               |
#        |      N      N      P      T      -      S      J      ,      . |
# -------+----------------------------------------------------------------+
#     NN | <10.6%>  0.1%   0.2%   0.0%      .   0.0%   0.7%      .      . |
#     IN |      .  <9.3%>  0.0%      .      .      .   0.0%      .      . |
#    NNP |   0.2%   0.0%  <7.9%>  0.0%      .   0.1%   0.4%      .      . |
#     DT |   0.0%   0.1%   0.0%  <8.3%>     .      .      .      .      . |
# -NONE- |      .      .      .      .  <6.5%>     .      .      .      . |
#    NNS |   0.0%      .   0.0%      .      .  <5.9%>  0.0%      .      . |
#     JJ |   0.3%   0.0%   0.1%   0.0%      .   0.0%  <4.7%>     .      . |
#      , |      .      .      .      .      .      .      .  <5.1%>     . |
#      . |      .      .      .      .      .      .      .      .  <3.7%>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# ============================== Evaluate Fold: 2 ==============================
# using_tagger: bayes
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u',', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'FW', u'POS', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR'])
# test tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u',', u"''", u'VBP', u'VBN', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'POS', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'CD', u'-NONE-', u'EX', u'IN', u'WP$', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'WDT'])
#     Accuracy: 0.925354474767
#    Precision: 0.976744186047
#       Recall: 0.976744186047
#    F-Measure: 0.976744186047
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |                    N             N      N                      |
#        |      N      I      N      D      E      N      J      C        |
#        |      N      N      P      T      -      S      J      D      , |
# -------+----------------------------------------------------------------+
#     NN | <11.6%>  0.1%   0.2%   0.0%   0.0%   0.1%   0.6%   0.1%      . |
#     IN |   0.0%  <9.4%>  0.0%   0.0%      .   0.0%   0.0%      .      . |
#    NNP |   0.3%   0.0%  <7.1%>  0.0%      .   0.2%   0.4%   0.0%      . |
#     DT |   0.0%   0.0%      .  <7.7%>     .      .      .      .      . |
# -NONE- |      .      .      .      .  <6.6%>     .      .      .      . |
#    NNS |   0.0%      .   0.0%      .      .  <5.8%>     .      .      . |
#     JJ |   0.2%   0.0%   0.1%   0.0%      .   0.0%  <4.7%>  0.0%      . |
#     CD |   0.0%      .   0.0%      .   0.0%   0.0%   0.0%  <4.9%>     . |
#      , |      .      .      .      .      .      .      .      .  <4.8%>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# [Finished in 395.5s]

# HMM
# ============================== Evaluate Fold: 0 ==============================
# using_tagger: hmm
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u'VBN', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'RP', u'$', u'NN', u'FW', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'CD', u'-NONE-', u'EX', u'IN', u'WP$', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'SYM', u'UH'])
# test tag set: set([u'PRP$'])
#     Accuracy: 0.00823753055128
#    Precision: 1.0
#       Recall: 0.0222222222222
#    F-Measure: 0.0434782608696
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |             N                    N             N               |
#        |      N      N      I      D      E      J      N               |
#        |      N      P      N      T      -      J      S      ,      . |
# -------+----------------------------------------------------------------+
#     NN |     <.>     .      .      .      .      .      .      .      . |
#    NNP |      .     <.>     .      .      .      .      .      .      . |
#     IN |      .      .     <.>     .      .      .      .      .      . |
#     DT |      .      .      .     <.>     .      .      .      .      . |
# -NONE- |      .      .      .      .     <.>     .      .      .      . |
#     JJ |      .      .      .      .      .     <.>     .      .      . |
#    NNS |      .      .      .      .      .      .     <.>     .      . |
#      , |      .      .      .      .      .      .      .     <.>     . |
#      . |      .      .      .      .      .      .      .      .     <.>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# ============================== Evaluate Fold: 1 ==============================
# using_tagger: hmm
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'VB', u'POS', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'FW', u',', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'``', u'WRB', u'CC', u'LS', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR', u'UH'])
# test tag set: set([u'PRP$'])
#     Accuracy: 0.00823803900107
#    Precision: 1.0
#       Recall: 0.0222222222222
#    F-Measure: 0.0434782608696
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |                    N             N      N                      |
#        |      N      I      N      D      E      N      J               |
#        |      N      N      P      T      -      S      J      ,      . |
# -------+----------------------------------------------------------------+
#     NN |     <.>     .      .      .      .      .      .      .      . |
#     IN |      .     <.>     .      .      .      .      .      .      . |
#    NNP |      .      .     <.>     .      .      .      .      .      . |
#     DT |      .      .      .     <.>     .      .      .      .      . |
# -NONE- |      .      .      .      .     <.>     .      .      .      . |
#    NNS |      .      .      .      .      .     <.>     .      .      . |
#     JJ |      .      .      .      .      .      .     <.>     .      . |
#      , |      .      .      .      .      .      .      .     <.>     . |
#      . |      .      .      .      .      .      .      .      .     <.>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# ============================== Evaluate Fold: 2 ==============================
# using_tagger: hmm
#  ori tag set: set([u'PRP$', u'VBG', u'VBD', u'``', u',', u"''", u'VBP', u'WDT', u'JJ', u'WP', u'VBZ', u'DT', u'#', u'RP', u'$', u'NN', u'FW', u'POS', u'.', u'TO', u'PRP', u'RB', u'-LRB-', u':', u'NNS', u'NNP', u'VB', u'WRB', u'CC', u'PDT', u'RBS', u'RBR', u'VBN', u'-NONE-', u'EX', u'IN', u'WP$', u'CD', u'MD', u'NNPS', u'-RRB-', u'JJS', u'JJR'])
# test tag set: set([u'PRP$'])
#     Accuracy: 0.0063119377954
#    Precision: 1.0
#       Recall: 0.0232558139535
#    F-Measure: 0.0454545454545
#        |                                  -                             |
#        |                                  N                             |
#        |                                  O                             |
#        |                    N             N      N                      |
#        |      N      I      N      D      E      N      J      C        |
#        |      N      N      P      T      -      S      J      D      , |
# -------+----------------------------------------------------------------+
#     NN |     <.>     .      .      .      .      .      .      .      . |
#     IN |      .     <.>     .      .      .      .      .      .      . |
#    NNP |      .      .     <.>     .      .      .      .      .      . |
#     DT |      .      .      .     <.>     .      .      .      .      . |
# -NONE- |      .      .      .      .     <.>     .      .      .      . |
#    NNS |      .      .      .      .      .     <.>     .      .      . |
#     JJ |      .      .      .      .      .      .     <.>     .      . |
#     CD |      .      .      .      .      .      .      .     <.>     . |
#      , |      .      .      .      .      .      .      .      .     <.>|
# -------+----------------------------------------------------------------+
# (row = reference; col = test)

# ==============================================================================
# [Finished in 268.8s]
