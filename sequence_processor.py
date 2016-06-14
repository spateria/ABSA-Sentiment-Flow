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



#from MPQAprecessor import MPQAsenti, MPQA_neg_list, MPQA_pos_list, 
from MPQAprecessor import bliu_pos, bliu_neg
                  
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



class TextClass:

    def __init__(self,sentence,frm,to,sentence2,frm2,to2,sentence3,frm3,to3,idn,prev_pol,prev_pol2,prev_idn):
        self.sentence = sentence 
        self.sentence2 = sentence2 
        self.sentence3 = sentence3 
        self.sentenceClean = ""

        self.frm = int(frm)
        self.to = int(to)
        self.frm2 = int(frm2)
        self.to2 = int(to2)
        self.frm3 = int(frm3)
        self.to3 = int(to3)
        
        self.idn = idn
        self.strBefore = ""
        self.strAfter = ""
        self.strBeforeClean = ""
        self.strAfterClean = ""
        self.sentenceLemma = ""
        self.strAfterLemma = ""
        self.sentenceLemma = ""
        self.prev_pol = prev_pol
        self.prev_pol2 = prev_pol2
        self.prev_idn = prev_idn
        #self.pol_arr = pol_arr
        
        
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
        
        
    def targetLocalStrings(self, sent, frm, to):
        #features around the target (from, to)
        #modify senti functions if they are to be used here
        
        strBefore = ""
        strAfter = ""
                    
        l = len(sent)
        
        if (sent != "") and ((frm != 0) or (to != 0)):
            if (frm-1)>=0:
                strBefore = sent[:frm]
            if (to+1)<l:
                strAfter = sent[(to+1):]
            
        self.strBefore = strBefore
        self.strAfter = strAfter
        
        #print self.strBefore, ' ', sent[frm:(to+1)], ' ', self.strAfter
        ############# LIMIT LENGTH HERE FOR LOCALITY ##########################################
      
        
    def ProcessSents(self, cnt):
        
        if cnt == 0:
            self.targetLocalStrings(self.sentence, self.frm, self.to) ### make before after FULL strings
        elif cnt == 1:
            self.targetLocalStrings(self.sentence2, self.frm2, self.to2)
        elif cnt == 2:
            self.targetLocalStrings(self.sentence3, self.frm3, self.to3)
        ##### LINIT LOCAL LENGHT FO RTHIS ##################################################
        
        ###CLEAN strings####
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
        
    def targetLocalGrams(self, sb, sa, cnt):
        D = {}
        
        ###################################################################
        uniBefore1=uniBefore2= biBefore1= biBefore2= triBefore= uniAfter1= uniAfter2= biAfter1= biAfter2= triAfter = ""
        
        strBefore = sb
        strAfter = sa 
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
            D["uniBefore_"+str(i)+str(cnt)] = uniBefore[i]
            
        D["biBefore1_"+str(cnt)] = biBefore1
        D["biBefore2_"+str(cnt)] = biBefore2
        #D["triBefore"] = triBefore
        
        ua = len(uniAfter)
        for i in range(ua):
            D["uniAfter_"+str(i)+str(cnt)] = uniAfter[i]
            
        D["biAfter1_"+str(cnt)] = biAfter1
        D["biAfter2_"+str(cnt)] = biAfter2
        #D["triAfter"] = triAfter
        
        #print uniBefore1, uniBefore2, biBefore1, biBefore2, triBefore, self.target
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
	
    def sentiWordNetFeat(self, cnt):
        
        D = {}
        
        #0: num_pos_tokens ; 1: num_neg_tokens ; 2: maximal_sent ; 3: posSum ; 4: negSum
        
        #sentenceLevelSW = self.sentiWordNetProc(self.sentenceClean)
        
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
        D["tb_num_pos"+str(cnt)] =  targetBeforeSW[0]
        D["tb_num_neg"+str(cnt)] = targetBeforeSW[1]
        D["tb_max_senti"+str(cnt)] = targetBeforeSW[2]
        D["tb_pos_sum"+str(cnt)] = targetBeforeSW[3]
        D["tb_neg_sum"+str(cnt)] = targetBeforeSW[4]
        
        D["ta_num_pos"+str(cnt)] =  targetAfterSW[0]
        D["ta_num_neg"+str(cnt)] = targetAfterSW[1]
        D["ta_max_senti"+str(cnt)] = targetAfterSW[2]
        D["ta_pos_sum"+str(cnt)] = targetAfterSW[3]
        D["ta_neg_sum"+str(cnt)] = targetAfterSW[4]
        
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
        
    def bing_liuFeat(self, cnt):
        
        D = {}
        
        #0: pos 1: neg
        
        #sentenceLevelBL = self.bing_liuProc(self.sentenceClean)
        
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
        D["tb_bliu_pos"+str(cnt)] =  targetBeforeBL[0]
        D["tb_bliu_neg"+str(cnt)] = targetBeforeBL[1]

        
        D["ta_bliu_pos"+str(cnt)] =  targetAfterBL[0]
        D["ta_bliu_neg"+str(cnt)] = targetAfterBL[1]

        
        #print D
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
        
    def isNeutralFeat(self, cnt):
        
        D = {}
        
        #sentNeg = self.isNeutralProc(self.sentenceClean)
        
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
            D["sb_neut_weight_"+str(x)+str(cnt)] = sentTargetBefore[0][x]
        
        for x in range(len(sentTargetAfter[0])):
            D["sa_neut_weight_"+str(x)+str(cnt)] = sentTargetAfter[0][x]

        #D["sl_neut_weight_SCORE"] = sentNeg[1]
        D["sa_neut_weight_SCORE"+str(cnt)] = sentTargetBefore[1]
        D["sb_neut_weight_SCORE"+str(cnt)] = sentTargetAfter[1]
        print D
        return D
        
    def prevPol(self):
        D = {}
        
        x = ""
        y = ""
        if self.idn == self.prev_idn:
            x = self.prev_pol
            y = self.prev_pol2
        
        D["prev_pol"] = x
        D["prev_pol2"] = y
        return D
        
    def baselinefeatures(self):
        D = {}
    
        self.ProcessSents(0)
   
        ########### GRAMS ################
        D.update(self.targetLocalGrams(self.strBeforeClean, self.strAfterClean, 0)) # grams around target
        D.update(self.sentiWordNetFeat(0))
        D.update(self.bing_liuFeat(0))
        D.update(self.isNeutralFeat(0))
        D.update(self.prevPol())
       # return D
        
        self.ProcessSents(1) ### ----------- previous
   
        ########### GRAMS ################
        D.update(self.targetLocalGrams(self.strBeforeClean, self.strAfterClean, 1)) # grams around target ---- previous
        D.update(self.sentiWordNetFeat(1))
        D.update(self.bing_liuFeat(1))
        D.update(self.isNeutralFeat(1))
        #return D
        
        self.ProcessSents(2) ### ----------- previous
   
        ########### GRAMS ################
        D.update(self.targetLocalGrams(self.strBeforeClean, self.strAfterClean, 2)) # grams around target ---- previous
        D.update(self.sentiWordNetFeat(2))
        D.update(self.bing_liuFeat(2))
        D.update(self.isNeutralFeat(2))
        return D
        
      