########### THIS IS FOR RESTO DATA ################################

import xml.etree.ElementTree as ET
import nltk
from main_processor import *
import argparse
import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import cross_validation
from sklearn import dummy, tree, ensemble, svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
from sklearn import preprocessing
from nltk.corpus import brown
from nltk.probability import FreqDist
import numpy as np
import nltk.collocations
import collections
import matplotlib.pyplot as plt
from sklearn import preprocessing
from main_processor_2 import CorpusClass2
from main_processorLAPTOP import TextClassLAPTOP


#from main_classifier import read_data

global_path = 'E:/SEMEVAL/FROM_HOME_NEW/'
################################################################
train = global_path+'ABSA/archive/train1.xml'

test = global_path+'ABSA/archive(1)/EN_REST_SB1_TEST.xml'

testout = global_path+'ABSA/archive(1)/EN_REST_SB1_TEST_Copy.xml'
################################################################
xml = 0
root = 0
xml2 = 0
root2 = 0
GLOBAL_VEC = 0

def read_data2(data_file, TXT):

    data = defaultdict(list)

    global xml2
    global root2
    
    xml2 = ET.parse(data_file)
    root2 = xml2.getroot()
    for sentence in root2.iter('sentence'):
        oos = sentence.get('OutOfScope')
        if oos != "TRUE":
            idn = sentence.get('id')
            idn2 = []
            for x in idn:
                if x != ':':
                    idn2.append(x)
                else:
                    break
            idn2 = ''.join(idn2)
            #print idn
            #print idn2
            
            text = sentence.find('text').text
            TXT.append(text)
            aspects = sentence.find('Opinions')
            if aspects != None:
                aspects = aspects.findall('Opinion')
            if aspects != None:
                for aspect in aspects:
                    data[text].append((aspect.get('target'), aspect.get('category'), aspect.get('polarity'), aspect.get('from'), aspect.get('to'), idn2))
    
    #print data
    return data
#########################################################################################################################################################################


def read_data(data_file, TXT):

    data = defaultdict(list)

    xml = ET.parse(data_file)
    root = xml.getroot()
    for sentence in root.iter('sentence'):
        oos = sentence.get('OutOfScope')
        if oos != "TRUE":
            idn = sentence.get('id')
            idn2 = []
            for x in idn:
                if x != ':':
                    idn2.append(x)
                else:
                    break
            idn2 = ''.join(idn2)
            #print idn
            #print idn2
            
            text = sentence.find('text').text
            TXT.append(text)
            aspects = sentence.find('Opinions')
            if aspects != None:
                aspects = aspects.findall('Opinion')
            if aspects != None:
                for aspect in aspects:
                    data[text].append((aspect.get('target'), aspect.get('category'), aspect.get('polarity'), aspect.get('from'), aspect.get('to'), idn2))
    #print data
    return data
#########################################################################################################################################################################    
    
    
def modifyTestData(plabels):
    
    l = len(plabels)
    i = 0
    ######################################################
    for sentence in root2.iter('sentence'):
        oos = sentence.get('OutOfScope')
        if oos != "TRUE":
            
            text = sentence.find('text').text
            aspects = sentence.find('Opinions')
            if aspects != None:
                aspects = aspects.findall('Opinion')
            if aspects != None:
                for aspect in aspects:
                    if plabels[i] == 0:
                        aspect.set('polarity', 'negative')
                    elif plabels[i] == 1:
                        aspect.set('polarity', 'positive')
                    elif plabels[i] == 2:
                        aspect.set('polarity', 'neutral')
                        
                    i+=1
                    if i>(l-1):
                        break
        
        if i>(l-1):
            break
            
    xml2.write(testout)
    ####################################################################
    
    
def ReadProcTestData(features, labels):         
    test_TXT = []
    test_data = read_data2(test, test_TXT)
    #test = read_data(test)
    
    
    #test_TXT = test_TXT[:100]
    ##########################################   GLOBAL ##########################################

    #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(test_TXT)
    
    pl = negl = nutl = el = 0
    prev_pol = ""
    prev_idn = ""
    pol_arr = []
    
    for i in range(l):
        dlen = len(test_data[test_TXT[i]])
        
        for k in range(dlen):
            
            #global_features.append(txt_corp_feats[i])
            
            sentence = test_TXT[i] 
            target = test_data[test_TXT[i]][k][0]
            cat = test_data[test_TXT[i]][k][1]
            pol = 0
            frm = test_data[test_TXT[i]][k][3]
            to = test_data[test_TXT[i]][k][4]
            idn = test_data[test_TXT[i]][k][5]
            
            if idn == prev_idn:
                pol_arr.append(prev_pol)
            else:
                pol_arr = []
                
                
            t = TextClass(sentence,target,cat,pol,frm,to,idn,prev_pol,prev_idn,pol_arr)
                
            featuredicts.append(t.baselinefeatures())

    
    local_features = GLOBAL_VEC.transform(featuredicts).toarray()

    #print local_features
    print ()
    features.append(local_features)  

    
    
def ReadProcTestDataLabeled(features, labels):         
    train_TXT = []
    train_data = read_data2(testout, train_TXT)
    #test = read_data(test)
    
    
    #train_TXT = train_TXT[:100]
    ##########################################   GLOBAL ##########################################

    #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(train_TXT)
    
    pl = negl = nutl = el = 0
    prev_pol = ""
    prev_idn = ""
    pol_arr = []
    
    for i in range(l):
        dlen = len(train_data[train_TXT[i]])
        
        for k in range(dlen):
            
            #global_features.append(txt_corp_feats[i])
            
            sentence = train_TXT[i] 
            target = train_data[train_TXT[i]][k][0]
            cat = train_data[train_TXT[i]][k][1]
            pol = train_data[train_TXT[i]][k][2]
            frm = train_data[train_TXT[i]][k][3]
            to = train_data[train_TXT[i]][k][4]
            idn = train_data[train_TXT[i]][k][5]
            
            if idn == prev_idn:
                pol_arr.append(prev_pol)
            else:
                pol_arr = []
                
                
            if pol == 'positive' or pol == 'negative' or pol == "neutral":
                t = TextClass(sentence,target,cat,pol,frm,to,idn,prev_pol,prev_idn,pol_arr)
                
                featuredicts.append(t.baselinefeatures())
                if t.pol == 'positive':
                    labels.append(1)
                    pl += 1
                elif t.pol == 'negative':
                    labels.append(0)
                    negl += 1
                elif t.pol == "neutral":
                    labels.append(2)
                    nutl += 1
                else:
                    el += 1
                    
                prev_pol = t.pol
            prev_idn = idn
            
    print "counts ", pl, negl, nutl, el
    #sys.exit(0)
    
    vec = DictVectorizer()
    
    local_features = GLOBAL_VEC.transform(featuredicts).toarray()

    print ()
    features.append(local_features)  
    labels = np.array(labels)

def getAllTestFeats():
    
    test_features = []
    test_labels = []   
    
    ReadProcTestData(test_features, test_labels)
  
    #############################################################
    of = global_path+'ABSA/savedfeatsTEST/global/featSave.txt'
    of2 = global_path+'ABSA/savedfeatsTEST/global/labelSave.txt'
    f = open(of, 'w')
    np.save(of, test_features[0])
    f.close()
    f = open(of2, 'w')
    np.save(of2, test_labels)
    f.close()
    #############################################################
  
      
def getAllTestFeatsLabeled():
    
    test_features = []
    test_labels = []   
    
    ReadProcTestDataLabeled(test_features, test_labels)
  
    #############################################################
    of = global_path+'ABSA/savedfeatsTEST/global/featSaveLABELED.txt'
    of2 = global_path+'ABSA/savedfeatsTEST/global/labelSaveLABELED.txt'
    f = open(of, 'w')
    np.save(of, test_features[0])
    f.close()
    f = open(of2, 'w')
    np.save(of2, test_labels)
    f.close()
    #############################################################



def ReadProcTrainData(features, labels):         
    train_TXT = []
    train_data = read_data(train, train_TXT)
    #test = read_data(test)
    
    print train_data['Leon is an East Village gem: casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.']
    print train_data['Leon is an East Village gem: casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.'][1]
    print train_data['Leon is an East Village gem: casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.'][1][0]
    print train_data['Leon is an East Village gem: casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.'][1][1]
    print train_data['Leon is an East Village gem: casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.'][1][2]
    print train_data['Leon is an East Village gem: casual but hip, with well prepared basic French bistro fare, good specials, a warm and lively atmosphere.'][1][3]
    ############ split data before proceeding ###########
    
    
    #train_TXT = train_TXT[:100]
    ##########################################   GLOBAL ##########################################

    #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(train_TXT)
    
    pl = negl = nutl = el = 0
    prev_pol = ""
    prev_idn = ""
    pol_arr = []
    
    for i in range(l):
        dlen = len(train_data[train_TXT[i]])
        
        for k in range(dlen):
            
            #global_features.append(txt_corp_feats[i])
            
            sentence = train_TXT[i] 
            target = train_data[train_TXT[i]][k][0]
            cat = train_data[train_TXT[i]][k][1]
            pol = train_data[train_TXT[i]][k][2]
            frm = train_data[train_TXT[i]][k][3]
            to = train_data[train_TXT[i]][k][4]
            idn = train_data[train_TXT[i]][k][5]
            
            if idn == prev_idn:
                pol_arr.append(prev_pol)
            else:
                pol_arr = []
                
                
            if pol == 'positive' or pol == 'negative' or pol == "neutral":
                t = TextClass(sentence,target,cat,pol,frm,to,idn,prev_pol,prev_idn,pol_arr)
                
                featuredicts.append(t.baselinefeatures())
                if t.pol == 'positive':
                    labels.append(1)
                    pl += 1
                elif t.pol == 'negative':
                    labels.append(0)
                    negl += 1
                elif t.pol == "neutral":
                    labels.append(2)
                    nutl += 1
                else:
                    el += 1
                    
                prev_pol = t.pol
            prev_idn = idn
            
    print "counts ", pl, negl, nutl, el
    #sys.exit(0)
    
    vec = DictVectorizer()
    
    local_features = vec.fit_transform(featuredicts).toarray()
    
    global GLOBAL_VEC
    GLOBAL_VEC = vec
    #### append local and global features
    '''
    features = []
    for i in range(len(global_features)):
        features.append(np.concatenate((global_features[i], local_features[i])))
    
    features = np.asarray(features)
    '''
    #print local_features
    print ()
    features.append(local_features)  
    labels = np.array(labels)
    


def getAllTrainFeats():

    
    train_features = []
    train_labels = []   
    
    ReadProcTrainData(train_features, train_labels)
  
    #############################################################
    of = global_path+'ABSA/savedfeats/FORTEST__2/featSave.txt'
    of2 = global_path+'ABSA/savedfeats/FORTEST__2/labelSave.txt'
    #of = global_path+'ABSA/savedfeats/only_grams/featSave.txt'
    #of2 = global_path+'ABSA/savedfeats/only_grams/labelSave.txt'
    f = open(of, 'w')
    np.save(of, train_features[0])
    f.close()
    f = open(of2, 'w')
    np.save(of2, train_labels)
    f.close()
    #############################################################
    
    
    
    
def predOnTestDataModify(train_feat, train_labels, test_feat, maxent):
    
    TrainX_i = train_feat
    Trainy_i = train_labels
        
    TestX_i = test_feat

    maxent.fit(TrainX_i,Trainy_i)
    ypred_i = maxent.predict(TestX_i)
    
    of = global_path+'ABSA/savedfeats/FORTEST__2/predLabels.txt'
    f = open(of, 'w')
    np.save(of, ypred_i)
    f.close()
    modifyTestData(ypred_i)
    
    print("--")

    

def FeatandClassifyTESTSVM(sclr, mxnt):
    
        
    of = global_path+'ABSA/savedfeats/FORTEST__2/featSave.txt.npy'
    of2 = global_path+'ABSA/savedfeats/FORTEST__2/labelSave.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    train_features = np.load(f)
    f.close()
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    train_labels = np.load(f)
    f.close()    
    
    of = global_path+'ABSA/savedfeatsTEST/global/featSave.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    test_features = np.load(f)
    
    
    '''ofgram = global_path+'ABSA/savedfeats/featSaveReducedTrainGrams.txt.npy'
    f = open(ofgram, 'rb')
    f.seek(0)
    train_featuresgram = np.load(f)
    ofgram = global_path+'ABSA/savedfeatsTEST/featSaveReducedTrainGrams.txt.npy'
    f = open(ofgram, 'rb')
    f.seek(0)
    test_featuresgram = np.load(f)
    
    feats = []
    for i in range(len(train_features)):
        feats.append(np.concatenate((train_features[i], train_featuresgram[i])))
    train_features = np.asarray(feats)
    
    feats = []
    for i in range(len(test_features)):
        feats.append(np.concatenate((test_features[i], test_featuresgram[i])))
    test_features = np.asarray(feats)'''
    
    
    maxent = []
    maxent.append( svm.SVC(C=100, kernel='rbf', class_weight={0: 2, 2: 10}, gamma = 0.001) );
    
    features2 = train_features
    print 'LEN:  ', features2.shape
    featuresTST = test_features
    print 'LEN:  ', featuresTST.shape
    
    if sclr >= 0:
        scaler = []
        print "\n-------------------------- with Scaler ---------------------------  ", sclr
        scaler.append(preprocessing.RobustScaler(with_centering=False))
        scaler.append(preprocessing.Normalizer())
        scaler.append(preprocessing.StandardScaler())
    
        features2 = scaler[sclr].fit_transform(features2)
        featuresTST = scaler[sclr].transform(featuresTST)
        
    
    print 'LEN:  ', features2.shape
    print 'LEN:  ', featuresTST.shape
    
        
    predOnTestDataModify(features2, train_labels, featuresTST, maxent[mxnt])   
    
    


def FeatandClassifyTESTSVM_labelled(sclr, mxnt):
        
    of = global_path+'ABSA/savedfeats/FORTEST__2/featSave.txt.npy'
    of2 = global_path+'ABSA/savedfeats/FORTEST__2/labelSave.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    train_features = np.load(f)
    f.close()
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    train_labels = np.load(f)
    f.close()    
    
    of = global_path+'ABSA/savedfeatsTEST/global/featSaveLABELED.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    test_features = np.load(f)
    
    
    ofgram = global_path+'ABSA/savedfeats/featSaveReducedTrainGrams.txt.npy'
    f = open(ofgram, 'rb')
    f.seek(0)
    train_featuresgram = np.load(f)
    ofgram = global_path+'ABSA/savedfeatsTEST/featSaveReducedTrainGrams.txt.npy'
    f = open(ofgram, 'rb')
    f.seek(0)
    test_featuresgram = np.load(f)
    
    feats = []
    for i in range(len(train_features)):
        feats.append(np.concatenate((train_features[i], train_featuresgram[i])))
    train_features = np.asarray(feats)
    
    feats = []
    for i in range(len(test_features)):
        feats.append(np.concatenate((test_features[i], test_featuresgram[i])))
    test_features = np.asarray(feats)
    
    maxent = []
    maxent.append( svm.SVC(C=100, kernel='rbf', class_weight={0: 2, 2: 10}, gamma = 0.001) );
    
    features2 = train_features
    print 'LEN:  ', features2.shape
    featuresTST = test_features
    print 'LEN:  ', featuresTST.shape
    
    if sclr >= 0:
        scaler = []
        print "\n-------------------------- with Scaler ---------------------------  ", sclr
        scaler.append(preprocessing.RobustScaler(with_centering=False))
        scaler.append(preprocessing.Normalizer())
        scaler.append(preprocessing.StandardScaler())
    
        features2 = scaler[sclr].fit_transform(features2)
        featuresTST = scaler[sclr].transform(featuresTST)
        
        
        
    predOnTestDataModify(features2, train_labels, featuresTST, maxent[mxnt])   
        



getAllTrainFeats()
getAllTestFeats()
#test_TXT = []
#test_data = read_data2(test, test_TXT)

FeatandClassifyTESTSVM(0,0)

print '<<<<<<<<<< GONNA SLEEP FOR SOME TIME>>>>>>>>>>>>>>>>>>>'
import time
print ()
time.sleep(10) 
print ()
print '<<<<<<<<<<<<<<<<<<< NAP TIME OVER>>>>>>>>>>>>>>>>>>>>>>'

#getAllTestFeatsLabeled()
#FeatandClassifyTESTSVM_labelled(0,0)
#modifyData()
