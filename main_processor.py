import xml.etree.ElementTree as ET
import sys
from collections import defaultdict
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.collocations import *
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import product_reviews_1
from nltk.corpus import opinion_lexicon
from nltk.corpus import sentiwordnet as swn

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'them', 'their', 'they',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'have', 'has', 'had', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'into', 
     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
     'own', 'so', 'than', 's', 't', 'can', 'will', 'just', 'now', 'someone', 'one', 'mine', 'us', 'every', 'm', 'd', 'kinda', 'upon'])

class CorpusClass:
    
    def __init__(self,data,txt):
        self.data = data
        self.txt = txt
        self.txt_clean = []
        for i in range(len(txt)):
            self.txt_clean.append(txt[i])
        #print self.txt_clean
        #self.txt_unigram = self.txt_clean
        self.txt_bigram = []
        self.txt_trigram = []
    '''  
    def getParams():
        l = len(txt)
        for i in range(len(TXT)):
        print train_data[TXT[0]][0][1]
    '''

    def clean_corpus(self):
        
        l = len(self.txt)
        
        for i in range(l):
            self.txt_clean[i] = re.sub("[^a-zA-Z]", " ", self.txt_clean[i])   ####CONSIDER KEEPING !, ? etc. symbols
            tbase = self.txt_clean[i].lower().split()
            t = [w for w in tbase if not w in stopwords]
            #t = [w for w in t if not nltk.pos_tag(nltk.word_tokenize(t)).startswith('N')] #Casa La Femme
            t = ' '.join( t )
            self.txt_clean[i] = t
        return self.txt_clean
        
    def POStag(self,word):
    
        t = nltk.word_tokenize(word)
        t = nltk.pos_tag(t)
            
        return t

    def get_wordnet_pos(self,treebank_tag):
    
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''
        
    def lemmatize(self):
        
        s = nltk.word_tokenize(self.sentence)
        
        for word in s:
            POS = POStag(word)
            w = wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(POS))
            
    
    def makeSimpleBOG(self, gram):
        vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 500) 
        
        if (gram == 1):
            count_features = vectorizer.fit_transform(self.txt_clean).toarray()
        if (gram == 2):
            count_features = vectorizer.fit_transform(self.txt_bigram).toarray()
        if (gram == 3):
            count_features = vectorizer.fit_transform(self.txt_trigram).toarray()
            #print vectorizer.get_feature_names()
            
        return count_features
        #print vectorizer.get_feature_names()
    
    def makeTrigram(self):
    
        l = len(self.txt_clean)

        for j in range(l):
            s = self.txt_clean[j]

            s = nltk.word_tokenize(s)
                                
            l2 = len(s)
            
            if l2 < 3:
                s.insert(0,'KORLEV')
                s.append('KORLEV')
                if l2 == 0:
                    s.append('KORLEV')


            T = []
            for i in range(l2):
                x = s[i] + '0' +s[i+1] + '0' + s[i+2]
                #x = ''.join(x)
                T.append(x)
                if i == (l2-3):
                    break
            
            T = ' '.join(T)
            self.txt_trigram.append(T)
            
        #return self.txt_trigram
        
    
    def makeBigram(self):
    
        l = len(self.txt_clean)

        for j in range(l):
            s = nltk.word_tokenize(self.txt_clean[j])
            
            l2 = len(s)
            if l2 < 2:
                s.insert(0,'KORLEV')
                s.append('KORLEV')
                    
            T = []
            for i in range(l2):
                x = s[i] + '0' +s[i+1]
                #x = ''.join(x)
                T.append(x)
                if i == (l2-2):
                    break
            
            T = ' '.join(T)
            self.txt_bigram.append(T)
            
        #return self.txt_bigram
    
    
            
            
########################################################################################################################################### 
###########################################################################################################################################   
from MPQAprecessor import MPQAsenti, MPQA_neg_list, MPQA_pos_list, bliu_pos, bliu_neg
                  
negation_list = set(['no', 'not', 'none', 'neither', 'never', 'don\'t', 'don', 'didn\'t','didn',
                    'wouldn\'t', 'wouldn', 'couldn\'t', 'couldn', 'isn\'t', 'isn', 'wasn\'t', 'wasn', 
                    'won\'t', 'nor', 'hardly' ])   # could be   ??????????
                    
connectors = set(['however', 'but', 'unless', 'though', 'tho'])
connectors2 = set(['however', 'but', 'though', 'tho', '!', '?', 'also'])
                    
neut_words_1 = set(['passable', 'average', 'normal', 'simple', 'okay', 'ok', 'not0great', 'nothing0great', 'not0extraordinary', 'not0outstanding', 'not0amazing', 'casual', 'fair', 'fine'])
neut_words_2 = set(['adequate', 'reasonable', 'recommend', 'moderate', 'typical','alright','overlook','overlooked','not0complain','just','either','relatively','relative','mind','quintessential','sort'])
neut_words_3 = set(['mediocre', 'mediorce', 'not0good', 'decent', 'expect', 'expectation', 'expected']) #### tend towards negative or positive
                    
neg_words = set(['frickin', 'tasteless', 'noisy', 'pricy', 'not0good', 'crowded', 'waste', 'rude',
                    'poor', 'overpriced', 'horrible', 'overrated', 'wait', 'mediocre'])

pos_words = set(['relatively0good', 'service0friendly', 'five0star', 'phenomenal', 'tasty', 'well', 'nice', 'good'])

## nothing0ordinary

class TextClass:

    def __init__(self,sentence,target,category,polarity,frm,to,idn,prev_pol,prev_idn,pol_arr):
        self.sentence = sentence 
        self.sentenceClean = ""
        self.target = target
        self.cat = category
        self.pol = polarity
        self.frm = int(frm)
        self.to = int(to)
        self.idn = idn
        self.strBefore = ""
        self.strAfter = ""
        self.strBeforeClean = ""
        self.strAfterClean = ""
        self.sentenceLemma = ""
        self.strAfterLemma = ""
        self.sentenceLemma = ""
        self.prev_pol = prev_pol
        self.prev_idn = prev_idn
        self.pol_arr = pol_arr

    def clean_sent(self, sentence):

        if sentence == "":
            return ""
            
        #regx = re.sub("[^a-zA-Z!?]", " ", sentence)   ####CONSIDER KEEPING !, ? etc. symbols
        regx = re.sub("[^a-zA-Z]", " ", sentence)
        tbase = regx.lower().split()
        t = [w for w in tbase if not w in stopwords]

        t = ' '.join( t )
        
        clean = t
        return clean

    def lemmatize(self, sentence):
        
        s = nltk.word_tokenize(sentence)
        lem_sent = []
        
        
        for word in s:
            t = nltk.word_tokenize(word)
            t = nltk.pos_tag(t)
            treebank_tag = t[0][1]
            if treebank_tag.startswith('J'):
                t = wordnet.ADJ
            elif treebank_tag.startswith('V'):
                t = wordnet.VERB
            elif treebank_tag.startswith('N'):
                t = wordnet.NOUN
            elif treebank_tag.startswith('R'):
                t = wordnet.ADV
            else:
                t = ''
            
            if t != '':
                w = wordnet_lemmatizer.lemmatize(word, t)
                lem_sent.append(w)
            else:
                w = wordnet_lemmatizer.lemmatize(word)
                lem_sent.append(w)
                                
        lem_sent = ' '.join(lem_sent)
        return lem_sent
        
        
        
    def CategoryFeature(self):
        D = {}
        #D['cat'] = self.cat
        #print D['cat']
        
        cE = ""
        cA = ""
        
        l = len(self.cat)
        for i in range(l):
            if self.cat[i] == '#':
                cE = self.cat[:i]
                cA = self.cat[(i+1):]
                
        #print cE, " ", cA     
        D["catE"] = cE
        D["catA"] = cA
        return D
        
    def TargetFeature(self):
        D = {}
        D['target'] = self.target
        return D  
    
    def targetLocalStrings(self):
        #features around the target (from, to)
        #modify senti functions if they are to be used here
        
        strBefore = ""
        strAfter = ""
        
        sent = self.sentence
        frm = self.frm
        to = self.to
        
        l = len(sent)
        
        if self.target != 'NULL':
            if (frm-1)>=0:
                strBefore = sent[:frm]
            if (to+1)<l:
                strAfter = sent[(to+1):]
            
            self.strBefore = strBefore
            self.strAfter = strAfter
        
        #print self.strBefore, ' ', sent[frm:(to+1)], ' ', self.strAfter
        ############# LIMIT LENGTH HERE FOR LOCALITY ##########################################
        
    
    def polsInReview(self):
         
        D = {}
        x = ""
        D["pol_prev_revgroup"] = 0   ##default
        
        #print self.prev_idn, '-----', self.idn
        '''
        if self.prev_idn != "":
            
            if self.idn == self.prev_idn:
                if self.prev_idn != "":
                    x = self.prev_pol
        '''
        #print 'prev id: ', self.prev_idn, 'my id: ', self.idn, 'prev_pol: ', self.prev_pol
        #print self.pol_arr
        l = len(self.pol_arr)
        
        base_scr = 0.7
        step = 0
        if l>0:
            step = float(1-base_scr)/l
        
        for i in range(l):
            x = self.pol_arr[i]
            wg = base_scr + (step * i)
            
            if x == 'positive':
                    D["pol_prev_revgroup"] += 1*wg   ############## should these be put separately???????
            elif x == 'negative':
                    D["pol_prev_revgroup"] += (-1)*wg
            elif x == "neutral":
                    D["pol_prev_revgroup"] += 0


        s = nltk.word_tokenize(self.sentenceClean)
        for w in s:
            if w in connectors:    ################# this is an intution: but, however etc. can invert sentiment
                if D["pol_prev_revgroup"] > 0: #### leave some gap between -ve and +ve
                    D["pol_prev_revgroup"] -= 0.2 * (D["pol_prev_revgroup"] - 0)
                elif D["pol_prev_revgroup"] < 0:
                    D["pol_prev_revgroup"] += 0.2 * (D["pol_prev_revgroup"] - 0)
                    
        #print D
        return D
                
                
    def targetLocalGrams(self):
        D = {}
        
        ###################################################################
        uniBefore1=uniBefore2= biBefore1= biBefore2= triBefore= uniAfter1= uniAfter2= biAfter1= biAfter2= triAfter = ""
        
        strBefore = self.strBeforeClean
        strAfter = self.strAfterClean 
        uniBefore = []
        uniAfter = []
        
        if strBefore != "":
            sb = nltk.word_tokenize(strBefore)
            lb = len(sb)
            
            for x in range(lb):
                #tag = nltk.pos_tag(nltk.word_tokenize(sb[x]))[0][1]
                #print 'tag', tag
                #if not tag.startswith('N'):
                uniBefore.append(sb[x])  ##################### HOW TO ASSIGN WEIGHTS????????
            
            if (lb-2) >= 0:
                biBefore1 = sb[lb-2] + '0' + sb[lb-1]
            if (lb-3) >= 0:
                biBefore2 = sb[lb-3] + '0' + sb[lb-2]
                triBefore = sb[lb-3] + '0' + sb[lb-2] + '0' + sb[lb-1]
            
            
        if strAfter != "":
            sa = nltk.word_tokenize(strAfter)
            la = len(sa)
            
            for x in range(la):
                #tag = nltk.pos_tag(nltk.word_tokenize(sa[x]))[0][1]
                #print tag
                #if not tag.startswith('N'):
                uniAfter.append(sa[x])

            if la > 1:
                biAfter1 = sa[0] + '0' + sa[1]
            if la > 2:
                biAfter2 = sa[1] + '0' + sa[2]
                triAfter = sa[0] + '0' + sa[1] + '0' + sa[2]
            
            
            
        ub = len(uniBefore)
        for i in range(ub):
            D["uniBefore"+str(i)] = uniBefore[i]
            
        D["biBefore1"] = biBefore1
        D["biBefore2"] = biBefore2
        #D["triBefore"] = triBefore
        
        ua = len(uniAfter)
        for i in range(ua):
            D["uniAfter"+str(i)] = uniAfter[i]
            
        D["biAfter1"] = biAfter1
        D["biAfter2"] = biAfter2
        #D["triAfter"] = triAfter
        
        #print uniBefore1, uniBefore2, biBefore1, biBefore2, triBefore, self.target
        print D
        return D
        
        
    def isNegationProc(self, sentence):
        ######################### ADD CONNECTORS ############################
        #D = {}
        if sentence == "":
            return ["",""]
            
        s = nltk.word_tokenize(sentence)
        l = len(s)
        
        neg_weight = ""
        conn_weight = ""
        
        for i in range(l):
            if s[i] in negation_list:
                neg_weight =  s[i] #see if increment should be lower --- also is weight really a good idea??
            if s[i] in connectors2:
                conn_weight = s[i]
        
       
        #print 'nw ', neg_weight
        D = []
        D.append(neg_weight)
        D.append(conn_weight)
        #print D
        return D
        
    def isNegationFeat(self):
        
        D = {}
        
        sentNeg = self.isNegationProc(self.sentenceClean)
        
        stb = nltk.word_tokenize(self.strBeforeClean)
        if len(stb) > 5:
            stb = stb[(len(stb)-5):]
        stb = ' '.join(stb)
        
        sta = nltk.word_tokenize(self.strAfterClean)
        if len(sta) > 5:
            sta = sta[:5]
        sta = ' '.join(sta)
        
        sentTargetBefore = self.isNegationProc(stb)
        sentTargetAfter = self.isNegationProc(sta)
        
        '''D["sl_neg_weight"] = sentNeg[0]
        D["sl_conn_weight"] = sentNeg[1]'''
        
        D["tb_neg_weight"] = sentTargetBefore[0]
        D["tb_conn_weight"] = sentTargetBefore[1]
        
        D["ta_neg_weight"] = sentTargetAfter[0]
        D["ta_conn_weight"] = sentTargetAfter[1]
        
        #print self.strBeforeClean, ' ' , self.target, ' ',  self.strAfterClean
        #if D["tb_neg_weight"] > 0:
            #print "tb_neg_weight ", D["tb_neg_weight"]
        
        print D
        return D
        
  
    def isNeutralProc(self, sentence):
        ######################### ADD CONNECTORS ############################
        #D = {}
        if sentence == "":
            return ["",0]
            
        s = nltk.word_tokenize(sentence)
        l = len(s)
        
        neut_weight = []
        score = 0
        cnt = 0
        
        a = 8
        b = 4
        c = 2
        
        for i in range(l):
            if s[i] in neut_words_1:
                neut_weight.append(s[i])
                score += a
                cnt += 1
            if (i+1)<l and (s[i] + '0' + s[i+1]) in neut_words_1:
                neut_weight.append(s[i] + '0' + s[i+1])
                score += a
                cnt += 1
            if (i-1)>=0 and (s[i-1] + '0' + s[i]) in neut_words_1:
                neut_weight.append(s[i-1] + '0' + s[i])
                score += a
                cnt += 1
            ######################################
            if s[i] in neut_words_2:
                neut_weight.append(s[i])
                score += b
                cnt += 1
            if (i+1)<l and (s[i] + '0' + s[i+1]) in neut_words_2:
                neut_weight.append(s[i] + '0' + s[i+1])
                score += b
                cnt += 1
            if (i-1)>=0 and (s[i-1] + '0' + s[i]) in neut_words_2:
                neut_weight.append(s[i-1] + '0' + s[i])
                score += b
                cnt += 1
             #########################################   
            if s[i] in neut_words_3:
                neut_weight.append(s[i])
                score += c
                cnt += 1
            if (i+1)<l and (s[i] + '0' + s[i+1]) in neut_words_3:
                neut_weight.append(s[i] + '0' + s[i+1])
                score += c
                cnt += 1
            if (i-1)>=0 and (s[i-1] + '0' + s[i]) in neut_words_3:
                neut_weight.append(s[i-1] + '0' + s[i])
                score += c
                cnt += 1
        
        #if cnt > 0:
            #score = float(score)/cnt
        #print 'nw ', neg_weight
        D = []
        D.append(neut_weight)
        D.append(score)
        return D
        
    def isNeutralFeat(self):
        
        D = {}
        
        sentNeg = self.isNeutralProc(self.sentenceClean)
        
        stb = nltk.word_tokenize(self.strBeforeClean)
        if len(stb) > 5:
            stb = stb[(len(stb)-5):]
        stb = ' '.join(stb)
        
        sta = nltk.word_tokenize(self.strAfterClean)
        if len(sta) > 5:
            sta = sta[:5]
        sta = ' '.join(sta)
        
        sentTargetBefore = self.isNeutralProc(stb)
        sentTargetAfter = self.isNeutralProc(sta)
        
        '''for x in range(len(sentNeg[0])):
            D["sl_neut_weight_"+str(x)] = sentNeg[0][x]'''
        
        for x in range(len(sentTargetBefore[0])):
            D["sb_neut_weight_"+str(x)] = sentTargetBefore[0][x]
        
        for x in range(len(sentTargetAfter[0])):
            D["sa_neut_weight_"+str(x)] = sentTargetAfter[0][x]

        #D["sl_neut_weight_SCORE"] = sentNeg[1]
        D["sa_neut_weight_SCORE"] = sentTargetBefore[1]
        D["sb_neut_weight_SCORE"] = sentTargetAfter[1]
        print D
        return D
        
            
    def bing_liuProc(self, sentence):
        
        #D = {}
        if sentence == "":
            D = [0,0]
            return D
            
        s = nltk.word_tokenize(sentence)
        l = len(s)
        
        pos_score = 0
        neg_score = 0
        
        for i in range(l):
            
            if s[i] in bliu_neg or s[i] in neg_words:
                neg_score += 1
                if (i-1) >= 0:
                    if s[i-1] in negation_list:
                            pos_score += 1
                            neg_score -= 1
                    elif (i-2) >= 0:
                        if s[i-1] in negation_list:
                                pos_score += 1
                                neg_score -= 1
                   
                                     
            if s[i] in bliu_pos or s[i] in pos_words:
                pos_score += 1
                if (i-1) >= 0:
                    if s[i-1] in negation_list:
                            pos_score -= 1
                            neg_score += 1
                    elif (i-2) >= 0:
                        if s[i-2] in negation_list:
                                pos_score -= 1
                                neg_score += 1
        
        D = []
        D.append(pos_score)
        D.append(neg_score)
        
        return D
        
    def bing_liuFeat(self):
        
        D = {}
        
        #0: pos 1: neg
        
        sentenceLevelBL = self.bing_liuProc(self.sentenceClean)
        
        stb = nltk.word_tokenize(self.strBeforeClean)
        if len(stb) > 5:
            stb = stb[(len(stb)-5):]
        stb = ' '.join(stb)
        
        sta = nltk.word_tokenize(self.strAfterClean)
        if len(sta) > 5:
            sta = sta[:5]
        sta = ' '.join(sta)
        
        targetBeforeBL = self.bing_liuProc(stb)
        targetAfterBL = self.bing_liuProc(sta)
        
        ####### SENTENCE LEVEL SENTI ###########################
        '''D["sl_bliu_pos"] =  sentenceLevelBL[0]
        D["sl_bliu_neg"] = sentenceLevelBL[1]'''
        
        
        ######### TARGET LEVEL SENTI ###########################
        D["tb_bliu_pos"] =  targetBeforeBL[0]
        D["tb_bliu_neg"] = targetBeforeBL[1]

        
        D["ta_bliu_pos"] =  targetAfterBL[0]
        D["ta_bliu_neg"] = targetAfterBL[1]

        
        #print D
        return D
        
        
    
    def MPQAProc(self, sentence):
        
        #D = {}
        if sentence == "":
            D = [0,0]
            return D
            
        s = nltk.word_tokenize(sentence)
        l = len(s)
        
        pos_score = 0
        neg_score = 0
        
        
        for i in range(l):
            if s[i] in MPQA_neg_list or s[i] in neg_words:
                neg_score += 1
                if (i-1) >= 0:
                    if s[i-1] in negation_list:
                            pos_score += 1
                            neg_score -= 1
                    elif (i-2) >= 0:
                        if s[i-1] in negation_list:
                                pos_score += 1
                                neg_score -= 1
                   
                                     
            if s[i] in MPQA_pos_list or s[i] in pos_words:
                pos_score += 1
                if (i-1) >= 0:
                    if s[i-1] in negation_list:
                            pos_score -= 1
                            neg_score += 1
                    elif (i-2) >= 0:
                        if s[i-2] in negation_list:
                                pos_score -= 1
                                neg_score += 1
        
        D = []
        D.append(pos_score)
        D.append(neg_score)
        
        return D
        
    def MPQAFeat(self):
        
        D = {}
        
        #0: pos 1: neg
        
        sentenceLevelBL = self.MPQAProc(self.sentenceClean)
        
        stb = nltk.word_tokenize(self.strBeforeClean)
        if len(stb) > 5:
            stb = stb[(len(stb)-5):]
        stb = ' '.join(stb)
        
        sta = nltk.word_tokenize(self.strAfterClean)
        if len(sta) > 5:
            sta = sta[:5]
        sta = ' '.join(sta)
        
        targetBeforeBL = self.MPQAProc(stb)
        targetAfterBL = self.MPQAProc(sta)
        
        ####### SENTENCE LEVEL SENTI ###########################
        '''D["sl_mpqa_pos"] =  sentenceLevelBL[0]
        D["sl_mpqa_neg"] = sentenceLevelBL[1]'''
        
        
        ######### TARGET LEVEL SENTI ###########################
        D["tb_mpqa_pos"] =  targetBeforeBL[0]
        D["tb_mpqa_neg"] = targetBeforeBL[1]

        
        D["ta_mpqa_pos"] =  targetAfterBL[0]
        D["ta_mpqa_neg"] = targetAfterBL[1]

        
        #print D
        return D
        
        
    def sentiWordNetProc(self, sentence):
        
        #D = {}
        
        if sentence == "":
            D = [0,0,0,0,0]
            return D
        
        s = nltk.word_tokenize(sentence)
        l = len(s)
        
	num_pos_tokens = 0
	num_neg_tokens = 0
	maximal_sentiment = 0
	sentimentScores = list()
	posSentimentSum = 0
	negSentimentSum = 0
	
	for i in range(l):
	    
	   senti_type = swn.senti_synsets(s[i])
	   if senti_type:
	        n = senti_type[0].neg_score()
	        p = senti_type[0].pos_score()
	        
	        if (i-1) >= 0:
                    if s[i-1] in negation_list:
                        p = float(n-p)/2 + p
                        n = float(p-n)/2 + n
                    elif (i-2) >= 0:
                        if s[i-2] in negation_list:
                            p = float(n-p)/2 + p
                            n = float(p-n)/2 + n
	        
	        
		posSentimentSum+=float(p)
		negSentimentSum+=float(n)
		
		if float(p)>float(n):
			num_pos_tokens+=1
			sentimentScores.append(float(p))
		else:
			num_neg_tokens+=1
			sentimentScores.append(float(n))
	if len(sentimentScores)>0:
		maximal_sentiment = max(sentimentScores)
	else:
		maximal_sentiment = 0
		
	D = []
	D.append(num_pos_tokens)
	D.append(num_neg_tokens)
	D.append(maximal_sentiment)
	D.append(posSentimentSum)
	D.append(negSentimentSum)
	
	return D
	
    def sentiWordNetFeat(self):
        
        D = {}
        
        #0: num_pos_tokens ; 1: num_neg_tokens ; 2: maximal_sent ; 3: posSum ; 4: negSum
        
        sentenceLevelSW = self.sentiWordNetProc(self.sentenceClean)
        
        stb = nltk.word_tokenize(self.strBeforeClean)
        if len(stb) > 5:
            stb = stb[(len(stb)-5):]
        stb = ' '.join(stb)
        
        sta = nltk.word_tokenize(self.strAfterClean)
        if len(sta) > 5:
            sta = sta[:5]
        sta = ' '.join(sta)
        
        targetBeforeSW = self.sentiWordNetProc(stb)
        targetAfterSW = self.sentiWordNetProc(sta)
        
        ####### SENTENCE LEVEL SENTI ###########################
        '''D["sl_num_pos"] =  sentenceLevelSW[0]
        D["sl_num_neg"] = sentenceLevelSW[1]
        D["sl_max_senti"] = sentenceLevelSW[2]
        D["sl_pos_sum"] = sentenceLevelSW[3]
        D["sl_neg_sum"] = sentenceLevelSW[4]'''
        
        
        ######### TARGET LEVEL SENTI ###########################
        D["tb_num_pos"] =  targetBeforeSW[0]
        D["tb_num_neg"] = targetBeforeSW[1]
        D["tb_max_senti"] = targetBeforeSW[2]
        D["tb_pos_sum"] = targetBeforeSW[3]
        D["tb_neg_sum"] = targetBeforeSW[4]
        
        D["ta_num_pos"] =  targetAfterSW[0]
        D["ta_num_neg"] = targetAfterSW[1]
        D["ta_max_senti"] = targetAfterSW[2]
        D["ta_pos_sum"] = targetAfterSW[3]
        D["ta_neg_sum"] = targetAfterSW[4]
        
        print D
        return D
  
  
    def mySentiParser(self):
    
        D = {}
        
        sntn = nltk.word_tokenize(self.sentenceClean)
        
        targ = nltk.word_tokenize(self.target)
        
        l = len(sntn)
        
        slist = [] 
        s= -1
        nlist = [] 
        n= -1
        tlist = [] 
        t = -1
        clist = [] 
        c = -1
        
        tfound = 0
        tsentbefore = []
        tsentafter = []
        
        nlegacy = [] ##########this should keep neg words not in vicinity of senti or target
        slegacy = []
        
        for i in range(l):
            word = sntn[i]
            sfnd = 0
            tfnd = 0
            nfnd = 0
            cfnd = 0
            senti_type = swn.senti_synsets(sntn[i])
            
            if word in MPQA_pos_list or word in pos_words:
                slist.append(1)
                s+=1
                sfnd = 1   ################### a senti word found
                
            '''if word in opinion_lexicon.positive() or word in pos_words:
                if sfnd == 1:
                    slist[s] = 1
                else:
                    slist.append(1)
                    s+=1
                    sfnd = 1'''
                    
            if senti_type and senti_type[0].pos_score() > 0:        
                if sfnd == 1:
                    slist[s] += senti_type[0].pos_score()
                    slist[s] = float(slist[s])/2
                else:
                    slist.append(float(1+senti_type[0].pos_score())/2)
                    s+=1
                    sfnd = 1
                    
                    
            if word in MPQA_neg_list or word in neg_words:
                slist.append(-1)
                s+=1
                sfnd = 1   ################### a senti word found
                
            '''if word in opinion_lexicon.negative():
                if sfnd == 1:
                    slist[s] = -1
                else:
                    slist.append(-1)
                    s+=1
                    sfnd = 1'''
                    
            if senti_type and senti_type[0].neg_score() > 0:        
                if sfnd == 1:
                    slist[s] -= senti_type[0].neg_score()
                    slist[s] = float(slist[s])/2
                else:
                    slist.append(float((-1)-senti_type[0].neg_score())/2)
                    s+=1
                    sfnd = 1
            
        
            if n >=0:    ###### a negative word was recently encountered
                if sfnd == 1:
                    slist[s] = slist[s] * (-1) ########### uhhh come on!!!!!!
                
                nlegacy.append(nlist)
                nlist = []
                n = -1
                
            if word in negation_list:
                nlist.append(1)
                n+=1
                nfnd = 1
                
                    
            #################### for sents before target ####################    
            if targ!='NULL' and (word in targ) and tfound!=1:
                ln = len(slist)
                for y in range(ln):
                    try:
                        tsentbefore.append( slist[y] * (0.6 + 0.4*float(y)/(ln-1)) )
                    except ZeroDivisionError:
                        print "So what!!!!"
                    
                slegacy.append(slist)
                slist = [] ######## all sents before target now do not matter
                s=-1
                tfound = 1
                
                    
                
        slegacy.append(slist)
        
        if tfound == 1:     ###########target was found when traversing############################
            ln = len(slist)
            for y in range(ln):
                try:
                    tsentafter.append( slist[y] * (1 - 0.4*float(y)/(ln-1)) )
                except ZeroDivisionError:
                        print "So what!!!!"
            slist = [] ######## all sents before target now do not matter
            s=-1
                
        
        nlegacy.append(nlist) #### cleaning
        
        
        ########scoring
        
        overallnegscore = 0
        for x in range(len(nlegacy)):
            overallnegscore += sum(nlegacy[x])
            
        overallsentscore = 0
        numneg = 0
        numpos = 0
        for x in range(len(slegacy)):
            overallsentscore += sum(slegacy[x])
            for i in range(len(slegacy[x])):
                if slegacy[x] > 0:
                    numpos += 1
                if slegacy[x] < 0:
                    numneg += 1
                    
        sentbef = sum(tsentbefore)
        sentaft = sum(tsentafter)
    
  
        D["overall_neg"] = overallnegscore
        D["overall_sent"] = overallsentscore
        D["numpos"] = numpos
        D["numneg"] = numneg
        D["sentbefore"] = sentbef
        D["sentafter"] = sentaft
        
        print D
        return D
        
        
    def ProcessSents(self):
        
        self.targetLocalStrings() ### make before after FULL strings
        ##### LINIT LOCAL LENGHT FO RTHIS ##################################################
        
        ###CLEAN strings####
        self.sentenceClean = self.clean_sent(self.sentence)
        self.strBeforeClean = self.clean_sent(self.strBefore)
        self.strAfterClean = self.clean_sent(self.strAfter)
        
        '''
        ### LEMMATIZE CLEANED STRING ####
        self.sentenceLemma = self.lemmatize(self.sentenceClean)
        self.strBeforeLemma = self.lemmatize(self.strBeforeClean)
        self.strAfterLemma = self.lemmatize(self.strAfterClean)
        
        print self.sentenceClean
        print self.sentenceLemma
        '''
        
    def baselinefeatures(self):
        D = {}
    
        self.ProcessSents()
        
        ##### DEGRADE #######
        #D.update(self.isNegationFeat()) #--------------> #Lone accuracy .70, F: .41   (increasing weight variance helps!!!!
        #D.update(self.CategoryFeature())  #--------------------------> #Accuracy degrader  Lone: .64, F .41
                                            #### AFFECT BOTH ACCURACY AND RECALL
                                            ### MAYBE USE AS REGULATOR LATER
        
        ########### GRAMS ################
        D.update(self.targetLocalGrams()) # grams around target
        
                
        #D.update(self.mySentiParser())
        
        ##### LEXICON######
        D.update(self.bing_liuFeat())  #bing li senti at sentence and target level
        D.update(self.sentiWordNetFeat())
        D.update(self.MPQAFeat())

        
        ### ADVANCED #####
        #D.update(self.polsInReview()) #-------< .82, .52
        D.update(self.isNeutralFeat()) #------> .67; .40 too many false +ve (when used with negation feat and prev pols 
                                                ## increases recall
        
        print 'KORLEV \n'
        return D
        

class DocClass:

    def __init__(self,sentence,target,category,polarity,frm,to):
        self.sentence = sentence 
        self.sentenceClean = ""
        self.target = target
        self.cat = category
        self.pol = polarity
        self.frm = int(frm)
        self.to = int(to)
        
        self.strBefore = ""
        self.strAfter = ""
        self.strBeforeClean = ""
        self.strAfterClean = ""


    def clean_sent(self, sentence):

        if sentence == "":
            return ""
            
        #regx = re.sub("[^a-zA-Z!?]", " ", sentence)   ####CONSIDER KEEPING !, ? etc. symbols
        regx = re.sub("[^a-zA-Z]", " ", sentence)
        tbase = regx.lower().split()
        t = [w for w in tbase if not w in stopwords]

        t = ' '.join( t )
        
        clean = t
        return clean

    def lemmatize(self, sentence):
        
        s = nltk.word_tokenize(sentence)
        lem_sent = []
        
        
        for word in s:
            t = nltk.word_tokenize(word)
            t = nltk.pos_tag(t)
            treebank_tag = t[0][1]
            if treebank_tag.startswith('J'):
                t = wordnet.ADJ
            elif treebank_tag.startswith('V'):
                t = wordnet.VERB
            elif treebank_tag.startswith('N'):
                t = wordnet.NOUN
            elif treebank_tag.startswith('R'):
                t = wordnet.ADV
            else:
                t = ''
            
            if t != '':
                w = wordnet_lemmatizer.lemmatize(word, t)
                lem_sent.append(w)
            else:
                w = wordnet_lemmatizer.lemmatize(word)
                lem_sent.append(w)
                                
        lem_sent = ' '.join(lem_sent)
        return lem_sent

    def targetLocalStrings(self):
        #features around the target (from, to)
        #modify senti functions if they are to be used here
        
        strBefore = []
        strAfter = []
        
        sent = self.sentence
        frm = self.frm
        to = self.to
        
        l = len(sent)
        
        if 1:
            if (frm-1)>=0:
                strB = sent[:frm]
                strBefore = nltk.word_tokenize(strB)
                if len(strBefore) >= 5:
                    strBefore = strBefore[:5]
                else:
                    strBefore = strBefore
                    
            if (to+1)<l:
                strA = sent[(to+1):]
                strAfter = nltk.word_tokenize(strA)
                if len(strAfter) >= 5:
                    strAfter = strAfter[:5]
                else:
                    strAfter = strAfter
            
        self.strBefore = ' '.join(strBefore)
        self.strAfter = ' '.join(strAfter)
        
        #print self.strBefore, ' ', sent[frm:(to+1)], ' ', self.strAfter
        ############# LIMIT LENGTH HERE FOR LOCALITY ##########################################
        
        
    def ProcessSents(self):
        
        self.targetLocalStrings() ### make before after FULL strings
        ##### LINIT LOCAL LENGHT FO RTHIS ##################################################
        
        ###CLEAN strings####
        #self.sentenceClean = self.clean_sent(self.sentence)
        self.strBeforeClean = self.clean_sent(self.strBefore)
        self.strAfterClean = self.clean_sent(self.strAfter)
        
        return self.strBeforeClean + ' ' + self.strAfterClean