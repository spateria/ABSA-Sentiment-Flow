import numpy as np
import sys
import nltk
'''
f = open('E:/SEMEVAL/MPQA.txt', 'r')
txt = f.readlines()

MPQAsenti = []

ln = len(txt)

for t in range(ln):
    
    x = nltk.word_tokenize(txt[t])
    
    a = ""
    for i in range(len(x[2])):
        if x[2][i] == '=':
            a = x[02][i+1:]
            break
    
    b = ""
    for i in range(len(x[5])):
        if x[5][i] == '=':
            b = x[5][i+1:]
            break
    
    word = a
    pol = b
    
    MPQAsenti.append([word,pol])
    #print MPQAsenti


MPQA_neg_list = []
MPQA_pos_list = []

for i in range(len(MPQAsenti)):
    if MPQAsenti[i][1] == 'positive':
        MPQA_pos_list.append(MPQAsenti[i][0])
    elif MPQAsenti[i][1] == 'negative':
        MPQA_neg_list.append(MPQAsenti[i][0])
    
    
print '\n\n'

#print MPQA_neg_list

print '\n'

#print MPQA_pos_list
        
#print MPQAsenti
'''
from nltk.corpus import opinion_lexicon

bliu_pos = opinion_lexicon.positive()
bliu_neg = opinion_lexicon.negative()


