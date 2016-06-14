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

        
            
########################################################################################################################################### 
###########################################################################################################################################            
negation_list = [ ['no', 1], ['not', 3], ['none', 1], ['neither', 1.5], ['never', 5], ['don\'t', 3], ['don', 3], ['didn\'t', 1.5],['didn', 1.5],
                    ['wouldn\'t', 1], ['wouldn', 1], ['couldn\'t', 1], ['couldn', 1], ['isn\'t', 1], ['isn', 1], ['wasn\'t', 2], ['wasn', 2], 
                    ['won\'t', 2], ['nor', 1.5], ['hardly', 2] ]    # could be   ??????????
                    
connectors = set(['however', 'but', 'unless', 'though', 'tho'])
connectors2 = set(['!', '?'])
                    
neut_words_1 = set(['passable', 'average', 'normal', 'simple', 'okay', 'ok', 'not0great', 'nothing0great', 'not0extraordinary', 'not0outstanding', 'not0amazing', 'casual', 'fair'])
neut_words_2 = set(['reasonable', 'recommend', 'moderate', 'typical','alright','overlook','overlooked','not0complain','just','either','relatively','relative','mind','quintessential','sort'])
neut_words_3 = set(['mediocre', 'mediorce', 'not0good', 'decent', 'expect', 'expectation', 'expected']) #### tend towards negative or positive
                    
neg_words = set(['frickin', 'tasteless', 'noisy', 'pricy', 'not0good', 'crowded', 'waste', 'rude',
                    'poor', 'overpriced', 'horrible'])

pos_words = set(['relatively0good', 'service0friendly', 'five0star', 'phenomenal', 'tasty', 'well'])

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
        
        base_scr = -500
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
                    D["pol_prev_revgroup"] -= 0.9 * (D["pol_prev_revgroup"] - 0)
                elif D["pol_prev_revgroup"] < 0:
                    D["pol_prev_revgroup"] += 0.9 * (D["pol_prev_revgroup"] - 0)
                    
        print D
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
            return [0,0]
            
        s = nltk.word_tokenize(sentence)
        l = len(s)
        
        neg_weight = 0
        conn_weight = 0
        
        for i in range(l):
            for x in range(len(negation_list)):
                if negation_list[x][0] == s[i]:
                    #print s[i]
                    neg_weight =  negation_list[x][1] * 2 #see if increment should be lower --- also is weight really a good idea??
                    #neg_weight = 1
            if s[i] in connectors2:
                conn_weight = 1
        
       
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
        
        D["sl_neg_weight"] = sentNeg[0]
        #D["sl_conn_weight"] = sentNeg[1]
        
        D["tb_neg_weight"] = sentTargetBefore[0]
        #D["tb_conn_weight"] = sentTargetBefore[1]
        
        D["ta_neg_weight"] = sentTargetAfter[0]
        #D["ta_conn_weight"] = sentTargetAfter[1]
        
        #print self.strBeforeClean, ' ' , self.target, ' ',  self.strAfterClean
        #if D["tb_neg_weight"] > 0:
            #print "tb_neg_weight ", D["tb_neg_weight"]
        
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
        
        for x in range(len(sentNeg[0])):
            D["sl_neut_weight_"+str(x)] = sentNeg[0][x]
        
        for x in range(len(sentTargetBefore[0])):
            D["sb_neut_weight_"+str(x)] = sentTargetBefore[0][x]
        
        for x in range(len(sentTargetAfter[0])):
            D["sa_neut_weight_"+str(x)] = sentTargetAfter[0][x]

        D["sl_neut_weight_SCORE"] = sentNeg[1]
        D["sa_neut_weight_SCORE"] = sentTargetBefore[1]
        D["sb_neut_weight_SCORE"] = sentTargetAfter[1]
        #print D
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
            
            if s[i] in opinion_lexicon.negative() or s[i] in neg_words:
                neg_score += 1
                if (i-1) >= 0:
                    if s[i-1] in negation_list:
                        if s[i-1] == 'never' or s[i-1] == 'don\'t':
                            pos_score += 1
                            neg_score -= 1
                        else:
                            pos_score += 1
                            neg_score -= 1
                    elif (i-2) >= 0:
                        if s[i-1] in negation_list:
                            if s[i-2] == 'never' or s[i-2] == 'don\'t':
                                pos_score += 1
                                neg_score -= 1
                            else:
                                pos_score += 1
                                neg_score -= 1
                   
                                     
            if s[i] in opinion_lexicon.positive() or s[i] in pos_words:
                pos_score += 1
                if (i-1) >= 0:
                    if s[i-1] in negation_list:
                        if s[i-1] == 'never' or s[i-1] == 'don\'t':
                            pos_score -= 1
                            neg_score += 1
                        else:
                            pos_score -= 1
                            neg_score += 1
                    elif (i-2) >= 0:
                        if s[i-2] in negation_list:
                            if s[i-2] == 'never' or s[i-2] == 'don\'t':
                                pos_score -= 1
                                neg_score += 1
                            else:
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
        D["sl_bliu_pos"] =  sentenceLevelBL[0]
        D["sl_bliu_neg"] = sentenceLevelBL[1]
        
        
        ######### TARGET LEVEL SENTI ###########################
        D["tb_bliu_pos"] =  targetBeforeBL[0]
        D["tb_bliu_neg"] = targetBeforeBL[1]

        
        D["ta_bliu_pos"] =  targetAfterBL[0]
        D["ta_bliu_neg"] = targetAfterBL[1]

        
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
                        if p>n:
                            p = float(p)/2
                            if n == 0:
                                n = p
                            else:
                                n = p + n*2
                        elif n>p:
                            n = float(n)/2
                            if p == 0:
                                p = n
                            else:
                                p = n
                    elif (i-2) >= 0:
                        if s[i-2] in negation_list:
                            if p>n:
                                p = float(p)/2
                                if n == 0:
                                    n = p
                                else:
                                    n = p + n*2
                            elif n>p:
                                n = float(n)/2
                                if p == 0:
                                    p = n
                                else:
                                    p = n
	        
	        
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
        D["sl_num_pos"] =  sentenceLevelSW[0]
        D["sl_num_neg"] = sentenceLevelSW[1]
        D["sl_max_senti"] = sentenceLevelSW[2]
        D["sl_pos_sum"] = sentenceLevelSW[3]
        D["sl_neg_sum"] = sentenceLevelSW[4]
        
        
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
        
        #print D
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
        
        ##### LEXICON######
        #D.update(self.bing_liuFeat())  #bing li senti at sentence and target level
        #D.update(self.sentiWordNetFeat())
        
        ### ADVANCED #####
        #D.update(self.polsInReview()) #-------< .82, .52
        #D.update(self.isNeutralFeat()) #------> .67; .40 too many false +ve (when used with negation feat and prev pols 
                                                ## increases recall
        
        
        return D
        
