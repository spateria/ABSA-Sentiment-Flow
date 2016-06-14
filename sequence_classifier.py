import xml.etree.ElementTree as ET
import nltk
from sequence_processor import *
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


global_path = 'F:/SENTI_EXPR/'
################################################################
train = global_path+'ABSA/archive/train1.xml'

test = global_path+'ABSA/archive(1)/EN_REST_SB1_TEST.xml'

testout = global_path+'ABSA/archive(1)/EN_REST_SB1_TEST_Copy.xml'
################################################################



# Here we will append index as well....this will be used to group parts of same review
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
    
    
def read_data(data_file, TXT):

    data = defaultdict(list)
    
    global xml
    global root
    
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
    

def ReadProcTrainData(train_features, labels):         
    train_TXT = []
    train_TXT_sec = []
    
    train_data = read_data(train, train_TXT)
    
    count = 0
     #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(train_TXT)
    
    
    prev_idn = ""
   
    
    c0 = c1 = c2 = 0    
    
    for i in range(l):
        dlen = len(train_data[train_TXT[i]])
        
        for k in range(dlen):
            count += 1
            #global_features.append(txt_corp_feats[i])
            
            sentence = train_TXT[i] 
            target = train_data[train_TXT[i]][k][0]
            cat = train_data[train_TXT[i]][k][1]
            pol = train_data[train_TXT[i]][k][2]
            frm = train_data[train_TXT[i]][k][3]
            to = train_data[train_TXT[i]][k][4]
            idn = train_data[train_TXT[i]][k][5]
            
            
            if idn != prev_idn:
                sentence2 = ""
                frm2 = 0
                to2 = 0
                sentence3 = ""
                frm3 = 0
                to3 = 0
                prev_pol = ""
                prev_pol2 = ""
                
            ''' train_TXT_sec.append(sentence)
                
                if pol == prev_pol:
                    labels.append(0)
                    c0 += 1
                elif pol == 'neutral':
                    print '------NEUT-----', train_TXT[i], '\n'
                    labels.append(1)     ## implies mild transform
                    c1 += 1
                else :
                    print '------NON -- NEUT-----', train_TXT[i], '\n'
                    labels.append(2)      ##implies hard transform
                    c2 += 1'''
            
            if pol == 'positive' or pol == 'negative' or pol == "neutral":
                t = TextClass(sentence,frm,to,sentence2,frm2,to2,sentence3,frm3,to3,idn,prev_pol,prev_pol2, prev_idn)
                featuredicts.append(t.baselinefeatures())
                
                if pol == 'positive':
                    labels.append(1)
                    c1 += 1
                elif pol == 'negative':
                    labels.append(0)
                    c0 += 1
                elif pol == "neutral":
                    labels.append(2)
                    c2 += 1
      
                
            
            prev_idn = idn
            
            prev_pol2 = prev_pol
            sentence3 = sentence2
            frm3 = frm2
            to3 = to2
            
            prev_pol = pol
            sentence2 = sentence
            frm2 = frm
            to2 = to
            
    
    #print train_TXT_sec, labels
    print len(train_TXT_sec), count
    
    vec = DictVectorizer()
    
    local_features = vec.fit_transform(featuredicts).toarray()
    
    global GLOBAL_VEC
    GLOBAL_VEC = vec
    #print local_features, '\n', len(local_features)
    print c0, c1, c2
    
    ################################################################################################################################
    #Work on sec. features 
    ############################################################################################################################
    
    print ()
    train_features.append(local_features)  
    labels = np.array(labels)
                

def getAllTrainFeats():
  
    train_features = []
    train_labels = []   
    
    ReadProcTrainData(train_features, train_labels)
  
    print len(train_features[0]), len(train_labels)
    
    #############################################################
    of = global_path+'ABSA/savedfeats/SEQTEST/featSave.txt'
    of2 = global_path+'ABSA/savedfeats/SEQTEST/labelSave.txt'
    #of = global_path+'ABSA/savedfeats/only_grams/featSave.txt'
    #of2 = global_path+'ABSA/savedfeats/only_grams/labelSave.txt'
    f = open(of, 'w')
    np.save(of, train_features[0])
    f.close()
    f = open(of2, 'w')
    np.save(of2, train_labels)
    f.close()
    #############################################################                
                                                


def ReadProcTestData(features, labels):         
    train_TXT = []
    
    train_data = read_data2(test, train_TXT)
    
    count = 0
     #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(train_TXT)
    
    
    prev_idn = ""
   
    
    c0 = c1 = c2 = 0    
    
    for i in range(l):
        dlen = len(train_data[train_TXT[i]])
        
        for k in range(dlen):
            count += 1
            #global_features.append(txt_corp_feats[i])
            
            sentence = train_TXT[i] 
            target = train_data[train_TXT[i]][k][0]
            cat = train_data[train_TXT[i]][k][1]
            pol = 0
            frm = train_data[train_TXT[i]][k][3]
            to = train_data[train_TXT[i]][k][4]
            idn = train_data[train_TXT[i]][k][5]
            
            
            if idn != prev_idn:
                sentence2 = ""
                frm2 = 0
                to2 = 0
                sentence3 = ""
                frm3 = 0
                to3 = 0
                prev_pol = ""
                
            
            t = TextClass(sentence,frm,to,sentence2,frm2,to2,sentence3,frm3,to3,idn,prev_pol,prev_idn)
            featuredicts.append(t.baselinefeatures())

            prev_pol = pol
            prev_idn = idn
            sentence3 = sentence2
            frm3 = frm2
            to3 = to2
            sentence2 = sentence
            frm2 = frm
            to2 = to
            

    
    local_features = GLOBAL_VEC.transform(featuredicts).toarray()

    #print local_features
    print ()
    features.append(local_features)  
    
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
    
    
def ReadProcTestDataLabeled(features, labels):         
    train_TXT = []
    train_TXT_sec = []
    
    train_data = read_data2(testout, train_TXT)
    
    count = 0
     #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(train_TXT)
    
    
    prev_idn = ""
   
    
    c0 = c1 = c2 = 0    
    
    for i in range(l):
        dlen = len(train_data[train_TXT[i]])
        
        for k in range(dlen):
            count += 1
            #global_features.append(txt_corp_feats[i])
            
            sentence = train_TXT[i] 
            target = train_data[train_TXT[i]][k][0]
            cat = train_data[train_TXT[i]][k][1]
            pol = train_data[train_TXT[i]][k][2]
            frm = train_data[train_TXT[i]][k][3]
            to = train_data[train_TXT[i]][k][4]
            idn = train_data[train_TXT[i]][k][5]
            
            
            if idn != prev_idn:
                sentence2 = ""
                frm2 = 0
                to2 = 0
                sentence3 = ""
                frm3 = 0
                to3 = 0
                prev_pol = ""
                
            ''' train_TXT_sec.append(sentence)
                
                if pol == prev_pol:
                    labels.append(0)
                    c0 += 1
                elif pol == 'neutral':
                    print '------NEUT-----', train_TXT[i], '\n'
                    labels.append(1)     ## implies mild transform
                    c1 += 1
                else :
                    print '------NON -- NEUT-----', train_TXT[i], '\n'
                    labels.append(2)      ##implies hard transform
                    c2 += 1'''
            
            if pol == 'positive' or pol == 'negative' or pol == "neutral":
                t = TextClass(sentence,frm,to,sentence2,frm2,to2,sentence3,frm3,to3,idn,prev_pol,prev_idn)
                featuredicts.append(t.baselinefeatures())
                
                if pol == 'positive':
                    labels.append(1)
                    c1 += 1
                elif pol == 'negative':
                    labels.append(0)
                    c0 += 1
                elif pol == "neutral":
                    labels.append(2)
                    c2 += 1
      
                
            prev_pol = pol
            prev_idn = idn
            sentence3 = sentence2
            frm3 = frm2
            to3 = to2
            sentence2 = sentence
            frm2 = frm
            to2 = to
            
    
    local_features = GLOBAL_VEC.transform(featuredicts).toarray()

    print ()
    features.append(local_features)  
    labels = np.array(labels)
    
    
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
    
    

    
    
def predOnTrainData(features, labels, maxent):
    
    features = np.asarray(features)
    labels = np.asarray(labels)

    for x in range(1):
        scores = defaultdict(list)
        scores2 = defaultdict(list)

        cnt = 0;
        for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
    
            TrainX_i = features[TrainIndices]
            Trainy_i = labels[TrainIndices]
    
            TestX_i = features[TestIndices]
            Testy_i =  labels[TestIndices]
    
            maxent.fit(TrainX_i,Trainy_i)
            ypred_i = maxent.predict(TestX_i)
            
            scores["Accuracy"].append(accuracy_score(Testy_i, ypred_i))
            scores["F1"].append(f1_score(Testy_i, ypred_i, average='macro'))
            scores["Precision"].append(precision_score(Testy_i, ypred_i, average='macro'))
            scores["Recall"].append(recall_score(Testy_i, ypred_i, average='macro'))

            cm = confusion_matrix(Testy_i, ypred_i)
            print(cm)
            cnt += 1

            from sklearn.metrics import precision_recall_fscore_support as score

            precision, recall, fscore, support = score(Testy_i, ypred_i)

            print('precision: {}'.format(precision))
            print('recall: {}'.format(recall))
            print('fscore: {}'.format(fscore))
            print('support: {}'.format(support))
    
        #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
        print("--")
    
        for key in sorted(scores.keys()):
            currentmetric = np.array(scores[key])
            print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
        print("\n--")
        
        

#### AKTSKI #######  
def FitandPredictSVM(sclr, mxnt):
    from sklearn.neighbors import KNeighborsClassifier
    of = global_path+'ABSA/savedfeats/SEQTEST/featSave.txt.npy'
    of2 = global_path+'ABSA/savedfeats/SEQTEST/labelSave.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    train_features = np.load(f)
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    train_labels = np.load(f)
    f.close()

    
    feat_backup = train_features   
    
    
    maxent = []

    maxent.append( svm.SVC(C=100, kernel='rbf', class_weight={0: 2, 2: 10}, gamma = 0.001) );

    maxent.append(KNeighborsClassifier(n_neighbors = 100))
    maxent.append(tree.DecisionTreeClassifier(class_weight={0: 2, 2: 10}))
    maxent.append(ensemble.RandomForestClassifier())
    maxent.append(ensemble.ExtraTreesClassifier())
    from sklearn.ensemble import AdaBoostClassifier
    maxent.append(AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(class_weight={0: 2, 2: 10}), n_estimators=100))
    from sklearn.ensemble import GradientBoostingClassifier
    maxent.append(GradientBoostingClassifier(n_estimators=1000))
    from sklearn.ensemble import VotingClassifier
    #maxent.append(VotingClassifier(estimators=[('svm', maxent[0]), ('dt', maxent[5])], voting='hard'))
    #maxent = svm.NuSVC(nu=0.5, kernel='rbf', class_weight='balanced');
    
    
    
    features2 = train_features
    print 'LEN:  ', features2.shape

    
    '''from sklearn.feature_selection import SelectKBest, chi2
    selection = SelectKBest(chi2, k='all')
    features2 = selection.fit(features2, train_labels).transform(features2)
    #features2_trial = selection.transform(features2_trial)'''
    print ()
    '''try:
        orig_stdout = sys.stdout
        f = file('D:/SEMEVAL/FROM_HOME_NEW/ABSA/rankedfeatsABSA.txt', 'w')
        sys.stdout = f
        top_ranked_features = sorted(enumerate(selection.scores_),key=lambda x:x[1], reverse=True)
        top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
        for feature_pvalue in zip(np.asarray(GLOBAL_VEC.get_feature_names())[top_ranked_features_indices],selection.pvalues_[top_ranked_features_indices]):
            print feature_pvalue
        sys.stdout = orig_stdout
        f.close()
    except NameError:
        print "well, it WASN'T defined after all!"'''
    
    if sclr >= 0:
        scaler = []
        print "\n-------------------------- with Scaler ---------------------------  ", sclr
        scaler.append(preprocessing.RobustScaler(with_centering=False))
        scaler.append(preprocessing.Normalizer())
        scaler.append(preprocessing.StandardScaler())
    
        features2 = scaler[sclr].fit_transform(features2)
        #features2_trial = scaler[sclr].transform(features2_trial)


    predOnTrainData(features2, train_labels, maxent[mxnt])
    
    print '=================== on trial ===================='
    
    #predOnTrial(features2, train_labels, features2_trial, trial_labels, maxent[mxnt])
    
    print "-----------------------KORLEV SVM--------------------------\n"
    

def FeatandClassifyTESTSVM(sclr, mxnt):
    
        
    of = global_path+'ABSA/savedfeats/SEQTEST/featSave.txt.npy'
    of2 = global_path+'ABSA/savedfeats/SEQTEST/labelSave.txt.npy'
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
        
    of = global_path+'ABSA/savedfeats/SEQTEST/featSave.txt.npy'
    of2 = global_path+'ABSA/savedfeats/SEQTEST/labelSave.txt.npy'
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
       
#############################################################################################################################################    



getAllTrainFeats()
#getAllTestFeats()
#getAllTestFeatsLabeled()
print '-----------------------------------------------AKTSKI----------------------'
FitandPredictSVM(0,0)
#FeatandClassifyTESTSVM(0,0)
#FeatandClassifyTESTSVM_labelled(0,0)