#coding=utf8

import nltk

class jc_nltk():
    def __init__(self,stop_words_path='/'.join(__file__.split('/')[:-1])+"/english.stop",show_exception = False):
        self.stop_words = []
        if len(stop_words_path) != 0:
            self.__load_stop_words(stop_words_path)

    def normalizer(self,s,using_pos_lema=False):
        s = str(s)
        if len(s) == 0:
            return [""]
        else:
            norm = self.remove_stop_words( self.tokenizer( self.data_clean(s) ) )
            if using_pos_lema:
                norm = self.pos_lemmatizer( norm )
            else:
                norm = self.lemmatizer( norm )
            if len(norm) == 0:
                return [""]
            else:
                return norm

    def data_clean(self,s):
        s = str(s)
        s = s.replace('’',"'")
        s = s.replace('“','"')
        s = s.replace('”','"')
        # remove punctual
        punctual = ",.?;:!\"(){}[]&@"
        # punctual = ",.?;:!\'\"(){}[]"
        for i in range(0,len(punctual)):
            s = s.replace( punctual[i] , "" )

        s = s.replace('\r'," ")
        s = s.replace('\n'," ")
        s = s.replace('\\n'," ")
        return s

    def remove_stop_words(self,s):
        return [str(word) for word in s if str(word).lower() not in self.stop_words]

    def tokenizer(self,s):
        s = str(s)
        # return nltk.word_tokenize(s)
        from nltk.tokenize import WhitespaceTokenizer
        tok = WhitespaceTokenizer().tokenize(s.lower())
        return tok

    def stemming(self,lis):
        from nltk.stem import PorterStemmer
        from nltk.stem import LancasterStemmer
        stemmer1 = PorterStemmer()
        stemmer2 = LancasterStemmer()
        stem = []
        for ele in lis:
            ele = str(ele)
            # word = stemmer1.stem(ele)
            # word = stemmer2.stem(word)
            word = stemmer1.stem(ele)
            stem.append( word )
        return stem

    def lemmatizer(self,lis):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lem = []
        for ele in lis:
            ele = str(ele)
            word = lemmatizer.lemmatize(ele)
            lem.append( word )
        return lem

    def pos_lemmatizer(self,lis,pos=""):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lem = []
        excepttion_case = []
        if pos == "":
            for ele in lis:
                ele = str(ele)
                # pos tag table:
                # http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
                # print ele
                # print nltk.pos_tag([ele])
                p_tag = nltk.pos_tag([ele])
                p = p_tag[0][1][0:1].lower()
                if p == "j":
                    p = 'a'
                try:
                    word = lemmatizer.lemmatize(ele,p)
                except:
                    # print "exception lemmatize word:", ele, "(", p+",", p_tag[0][1], ")"
                    excepttion_case.append( [ele,p,p_tag[0][1]] )
                    word = lemmatizer.lemmatize(ele)
                lem.append( word )
        else:
            for ele in lis:
                ele = str(ele)
                word = lemmatizer.lemmatize(ele,pos)
                lem.append( word )
        # print excepttion_case
        return lem

    def __load_stop_words(self,path):
        with open(path) as op:
            for line in op:
                self.stop_words.append( line.replace("\n","").replace("\r","").lower() )

if __name__ == '__main__':

    jt = jc_nltk("english.stop")

    # print jt.nltk.pos_tag('apple')
    # print jt.nltk.pos_tag(['apple'])

    string = "apples apple personalization\rpersonalized\naltered ran went waits beautiful a \r \n don't"
    # print string
    # print jt.tokenizer( string )

    # print jt.stemming( jt.tokenizer( string ) )
    # print jt.lemmatizer( jt.tokenizer( string ) )
    # print jt.pos_lemmatizer( jt.tokenizer( string ),'n' )
    # print jt.pos_lemmatizer( jt.tokenizer( string ) )
    print jt.normalizer( string )
    print jt.normalizer( "g" )