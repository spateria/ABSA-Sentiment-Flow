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

global_path = 'E:/SEMEVAL/FROM_HOME_NEW/'
################################################################
train = global_path+'ABSA/archive/train1.xml'
train2 = global_path+'ABSA/archive(2)/ABSA16_Laptops_Train_SB1_v2.xml'

trial = global_path+'ABSA/archive/trial1.xml'


################################################################
xml = 0
root = 0
xml2 = 0
root2 = 0




############################################ read train data ######################################################################################################


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
    

def modifyTestData(plabels):
    
    l = len(plabels)
    i = 0
    ######################################################
    for sentence in root.iter('sentence'):
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
            
    xml.write(testout)
    ####################################################################
    
    
#########################################################################################################################################################################

def ReadProcTrainDataGlobal():
    
    train_TXT = []
    train_data = read_data(train, train_TXT)
    #sys.exit(0)
    myCorp = CorpusClass2(train_data, train_TXT)
    clean_TXT = myCorp.clean_corpus()
    lemma_TXT = myCorp.lemmatize()
    #print clean_TXT
    
    l1 = len(clean_TXT)
    l2 = len(lemma_TXT)
    
    if l1 == l2:
        print 'shabaash bhaijaan!!!!!!'
        
    '''
    myCorp.makeTrigram()
    myCorp.makeBigram()
    bigram_count_feats = myCorp.makeSimpleBOG(2)
    
    txt_corp_feats = bigram_count_feats
    print txt_corp_feats
    print 'txt corp feat len ', txt_corp_feats.shape
    global_features = []
    
    l = len(train_TXT)
    for i in range(l):
        dlen = len(train_data[train_TXT[i]])
        for k in range(dlen):
            global_features.append(txt_corp_feats[i])

    #############################################################
    of = global_path+'ABSA/savedfeats/featSaveGLOBAL.txt'
    f = open(of, 'w')
    np.save(of, global_features)
    f.close()
    ##################################################
    '''
    
    #############################################################
    of = global_path+'ABSA/savedfeats/lemma_sents/sentLemmas.txt'
    f = open(of, 'w')
    np.save(of, lemma_TXT)
    f.close()
    ##################################################
    
    
    ###################################################
    of = global_path+'ABSA/savedfeats/lemma_sents/sentLemmas.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    lemma_sents = np.load(f)
    f.close()
    #####################################################
    
    print lemma_sents
    
    '''
    orig_stdout = sys.stdout
    f = file(global_path+'ABSA/parsed_sents.txt', 'w')
    sys.stdout = f
    
    from StanfordParser import dependency_parser
    import time
    l = len(clean_TXT)
    count = 0
    
    for k in range(l):
        count +=1 
        
        #print clean_TXT[k]
        result = dependency_parser.raw_parse(clean_TXT[k])
        dep = result.next()
        l = len(list(dep.triples())[:][:])
        for x in range(l):
            print list(dep.triples())[:][:][x]
        #if count > 10:
            #break
            
        print "\n$$$$\n"
        
        time.sleep(10)
        
    sys.stdout = orig_stdout
    f.close()
    '''
    
    
    
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
    
    #print 'local feat len ', len(local_features)
    #print 'label len ', len(labels)
    #print features
    
    '''for x in labels:
        if x == 1:
            print x'''
            
    '''
    scaler = preprocessing.StandardScaler()
    features=scaler.fit_transform(features)
    features2=scaler.transform(features2)
    '''


def ReadProcTrainData2(features, labels):         
    train_TXT = []
    train_data = read_data(train2, train_TXT)
    
    
    #train_TXT = train_TXT[:100]
    ##########################################   GLOBAL ##########################################

    #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(train_TXT)
    
    pl = negl = nutl = el = 0
    prev_pol = ""
    prev_idn = ""
    
    for i in range(l):
        dlen = len(train_data[train_TXT[i]])
        
        for k in range(dlen):
            
            #global_features.append(txt_corp_feats[i])
            
            sentence = train_TXT[i] 
            target = train_data[train_TXT[i]][k][0]
            cat = train_data[train_TXT[i]][k][1]
            pol = train_data[train_TXT[i]][k][2]
            idn = train_data[train_TXT[i]][k][5]
            
            if pol == 'positive' or pol == 'negative' or pol == "neutral":
                t = TextClassLAPTOP(sentence,target,cat,pol,idn,prev_pol,prev_idn)
                
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
    
    global GLOBAL_VEC2
    GLOBAL_VEC2 = vec
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
    



def ReadProcTrialData(features, labels):
    trial_TXT = []
    trial_data = read_data(trial, trial_TXT)
    
    print trial_TXT

    #############################################################################################################################################
    
    featuredicts = []
    
    ##################################### LOCAL and APPEND #######################################################
    l = len(trial_TXT)
    
    pl = negl = nutl = el = 0
    
    
    for i in range(l):
        dlen = len(trial_data[trial_TXT[i]])
        
        for k in range(dlen):
            
            #global_features.append(txt_corp_feats[i])
            
            sentence = trial_TXT[i] 
            target = trial_data[trial_TXT[i]][k][0]
            cat = trial_data[trial_TXT[i]][k][1]
            pol = trial_data[trial_TXT[i]][k][2]
            frm = trial_data[trial_TXT[i]][k][3]
            to = trial_data[trial_TXT[i]][k][4]
            
            if pol == 'positive' or pol == 'negative' or pol == "neutral":
                t = TextClass(sentence,target,cat,pol,frm,to)
                
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
            
    print "counts ", pl, negl, nutl, el
    #sys.exit(0)
    
    local_features = GLOBAL_VEC.transform(featuredicts).toarray()
    
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
    
    #print 'local feat len ', len(local_features)
    #print 'label len ', len(labels)
    #print features
    


def getAllTrainFeats():

    
    train_features = []
    train_labels = []   
    
    ReadProcTrainData(train_features, train_labels)
  
    #############################################################
    of = global_path+'ABSA/savedfeats/FORTEST/featSave.txt'
    of2 = global_path+'ABSA/savedfeats/FORTEST/labelSave.txt'
    #of = global_path+'ABSA/savedfeats/only_grams/featSave.txt'
    #of2 = global_path+'ABSA/savedfeats/only_grams/labelSave.txt'
    f = open(of, 'w')
    np.save(of, train_features[0])
    f.close()
    f = open(of2, 'w')
    np.save(of2, train_labels)
    f.close()
    #############################################################
    

def getAllTrainFeats2():

    
    train_features = []
    train_labels = []   
    
    ReadProcTrainData2(train_features, train_labels)
  
    #############################################################
    of = global_path+'ABSA/savedfeats/only_prevs/featSave2.txt'
    of2 = global_path+'ABSA/savedfeats/only_prevs/labelSave2.txt'
    f = open(of, 'w')
    np.save(of, train_features[0])
    f.close()
    f = open(of2, 'w')
    np.save(of2, train_labels)
    f.close()
    #############################################################
   
    
      

def getAllTrialFeats():
      
    trial_features = []
    trial_labels = []
    
    ReadProcTrialData(trial_features, trial_labels)
    #sys.exit(0)
    #############################################################
    of = global_path+'ABSA/savedfeats/featSaveTRIAL.txt'
    of2 = global_path+'ABSA/savedfeats/labelSaveTRIAL.txt'
    f = open(of, 'w')
    np.save(of, trial_features[0])
    f.close()
    f = open(of2, 'w')
    np.save(of2, trial_labels)
    f.close()
    #############################################################
    
        
    
###################################################################################################################################################################################
def predOnTrainData(features, labels, maxent):
    
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    '''for i in range(len(features[0])):
        if features[0][i] < 0:
            print i
        
    x = 2513
    print features[x][72]
    print features[x][74]
    print features[x][129]
    print features[x][130]
    print features[x][131]
    print features[x][135]
    print features[x][140]'''


    for x in range(1):
        scores = defaultdict(list)
        scores2 = defaultdict(list)

        cnt = 0;
        for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
            '''l = len(features)
            TrainIndices = range(0,((3*l)/4))
            TestIndices = range(((3*l)/4), l)'''
    
            TrainX_i = features[TrainIndices]
            Trainy_i = labels[TrainIndices]
    
            TestX_i = features[TestIndices]
            Testy_i =  labels[TestIndices]
    
            maxent.fit(TrainX_i,Trainy_i)
            ypred_i = maxent.predict(TestX_i)
    
            '''for x in ypred_i:
                if x == 0:
                    print x'''
            
            #a = accuracy_score(Testy_i, ypred_i)
            #p = precision_score(Testy_i, ypred_i)
            #r = recall_score(Testy_i, ypred_i)
            
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
    
    
    
def predOnTrial(train_feat, train_labels, trial_feat, trial_labels, maxent):  
     
    scores = defaultdict(list)

    TrainX_i = train_feat
    Trainy_i = train_labels
        
    TestX_i = trial_feat
    Testy_i = trial_labels

    maxent.fit(TrainX_i,Trainy_i)
    ypred_i = maxent.predict(TestX_i)

    #for x in ypred_i:
        #if x == 0:
           # print x
    
    scores["Accuracy"].append(accuracy_score(Testy_i, ypred_i))
    scores["F1"].append(f1_score(Testy_i, ypred_i, average='macro'))
    scores["Precision"].append(precision_score(Testy_i, ypred_i, average='macro'))
    scores["Recall"].append(recall_score(Testy_i, ypred_i, average='macro'))
         
    cm = confusion_matrix(Testy_i, ypred_i)
    print(cm)

    print scores
    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("--")


#### AKTSKI #######  
def FitandPredictSVM(sclr, mxnt):
    from sklearn.neighbors import KNeighborsClassifier
    of = global_path+'ABSA/savedfeats/FORTEST/featSave.txt.npy'
    of2 = global_path+'ABSA/savedfeats/FORTEST/labelSave.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    train_features = np.load(f)
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    train_labels = np.load(f)
    f.close()
    

    '''ofgram = global_path+'ABSA/savedfeats/featSaveReducedTrainGrams.txt.npy'
    f = open(ofgram, 'rb')
    f.seek(0)
    train_featuresgram = np.load(f)
    
    feats = []
    for i in range(len(train_features)):
        feats.append(np.concatenate((train_features[i], train_featuresgram[i])))
    train_features = np.asarray(feats)'''
    
    '''of = global_path+'ABSA/savedfeats/featSaveTRIAL.txt.npy'
    of2 = global_path+'ABSA/savedfeats/labelSaveTRIAL.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    trial_features = np.load(f)
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    trial_labels = np.load(f)
    f.close()
    features2_trial = trial_features'''
    
    feat_backup = train_features
    '''feats = []
    for i in range(len(global_features)):
        feats.append(np.concatenate((global_features[i], train_features[i])))
    train_features = np.asarray(feats)'''
    
    
    maxent = []

    maxent.append( svm.SVC(C=100, kernel='rbf', class_weight={0: 2, 2: 10}, gamma = 0.001) );
    maxent.append( svm.SVC(C=100, kernel='rbf', class_weight={0: 2, 2: 15}, gamma = 0.001) );

    maxent.append(KNeighborsClassifier(n_neighbors = 100))
    maxent.append(tree.DecisionTreeClassifier(class_weight={0: 2, 2: 10}))
    maxent.append(ensemble.RandomForestClassifier())
    maxent.append(ensemble.ExtraTreesClassifier())
    from sklearn.ensemble import AdaBoostClassifier
    maxent.append(AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(class_weight={0: 2, 2: 10}), n_estimators=100))
    from sklearn.ensemble import GradientBoostingClassifier
    maxent.append(GradientBoostingClassifier(n_estimators=1000))
    from sklearn.ensemble import VotingClassifier
    maxent.append(VotingClassifier(estimators=[('svm', maxent[0]), ('svm2', maxent[1]), ('dt', maxent[3]), ('knn', maxent[2]),('rf', maxent[4])], voting='soft'))
    from sklearn.ensemble import BaggingClassifier
    maxent.append(BaggingClassifier(base_estimator = maxent[3], n_estimators = 100, bootstrap = True))
    #maxent = svm.NuSVC(nu=0.5, kernel='rbf', class_weight='balanced');
    
    
    
    features2 = train_features    ########there are too many unnecessary unigrams
    print 'LEN:  ', features2.shape
    #features2_trial = trial_features
    #print 'LEN:  ', features2_trial.shape
    
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

    
    

    
def FitandPredictSVM2(sclr, mxnt):
    
    from sklearn.neighbors import KNeighborsClassifier
    of = global_path+'ABSA/savedfeats/only_prevs/featSave2.txt.npy'
    ofbliu = global_path+'ABSA/savedfeats/only_bliu/featSave2.txt.npy'
    of2 = global_path+'ABSA/savedfeats/only_prevs/labelSave2.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    train_features = np.load(f)
    f.close()
    f = open(ofbliu, 'rb')
    f.seek(0)
    train_featuresbliu = np.load(f)
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    train_labels = np.load(f)
    f.close()
    
    '''
    of = global_path+'ABSA/savedfeats2/featSaveGLOBAL.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    global_features = np.load(f)
    f.close()
    '''
    '''of = 'D:/SEMEVAL/FROM_HOME_NEW/ABSA/savedfeats2/featSaveTRIAL.txt.npy'
    of2 = 'D:/SEMEVAL/FROM_HOME_NEW/ABSA/savedfeats2/labelSaveTRIAL.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    trial_features = np.load(f)
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    trial_labels = np.load(f)
    f.close()'''
    
    
    feat_backup = train_features
    feats = []
    for i in range(len(train_featuresbliu)):
        feats.append(np.concatenate((train_featuresbliu[i], train_features[i])))
    train_features = np.asarray(feats)
    
    '''feats = []
    for i in range(len(global_features)):
        feats.append(np.concatenate((global_features[i], train_features[i])))
    train_features = np.asarray(feats)'''
    
    
    maxent = []

    maxent.append( svm.SVC(C=100, kernel='rbf', class_weight={0: 2, 2: 10}, gamma = 0.001) );
    maxent.append( svm.SVC(C=500, kernel='rbf', class_weight={0: 2, 2: 10}, gamma = 0.01) );

    
    maxent.append( svm.SVC(C=70, kernel='rbf', class_weight={1: 1.8}, gamma = 0.001) )
    maxent.append( svm.SVC(C=70, kernel='rbf', class_weight={1: 2}, gamma = 0.1) )
    maxent.append( svm.SVC(C=70, kernel='rbf', class_weight={1: 1.8}, gamma = 0.1) )
    maxent.append(KNeighborsClassifier(n_neighbors = 100))
    maxent.append(tree.DecisionTreeClassifier(class_weight={0: 2, 2: 13}))
    maxent.append(ensemble.RandomForestClassifier())
    maxent.append(ensemble.ExtraTreesClassifier())
    from sklearn.ensemble import AdaBoostClassifier
    maxent.append(AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(class_weight={0: 2, 2: 13}), n_estimators=100))
    from sklearn.ensemble import GradientBoostingClassifier
    maxent.append(GradientBoostingClassifier(n_estimators=1000))
    from sklearn.ensemble import VotingClassifier
    maxent.append(VotingClassifier(estimators=[('svm', maxent[0]), ('dt', maxent[5])], voting='hard'))
    #maxent = svm.NuSVC(nu=0.5, kernel='rbf', class_weight='balanced');
    
    
    
    features2 = train_features
    print 'LEN:  ', features2.shape
    #features2_trial = trial_features
    #print 'LEN:  ', features2_trial.shape
    
    from sklearn.feature_selection import SelectKBest, chi2
    selection = SelectKBest(chi2, k='all')
    #features2 = selection.fit(features2, train_labels).transform(features2)
    #features2_trial = selection.transform(features2_trial)
    print ()
    '''try:
        orig_stdout = sys.stdout
        f = file(global_path+'ABSA/rankedfeatsABSA2.txt', 'w')
        sys.stdout = f
        top_ranked_features = sorted(enumerate(selection.scores_),key=lambda x:x[1], reverse=True)
        top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
        for feature_pvalue in zip(np.asarray(GLOBAL_VEC2.get_feature_names())[top_ranked_features_indices],selection.pvalues_[top_ranked_features_indices]):
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
    
        
            
                
                        
def GridSearchOnData(features, labels):
    
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.cross_validation import train_test_split
    
    '''X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.75, random_state=0)'''

    tuned_parameters = [{'kernel': ['rbf'], 'class_weight': [None, {1: 1.5}], 'gamma': [ 1, 0.1, 0.01 , 1e-3, 1e-4],
                     'C': [1, 5, 10, 100, 1000]}]


    scaler = preprocessing.RobustScaler(with_centering=False)
    features2 = scaler.fit_transform(features)
    
    scores = ['recall', 'accuracy']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                        scoring='%s' % score)
        #clf.fit(X_train, y_train)
        clf.fit(features2, labels)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                % (mean_score, scores.std() * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        #y_true, y_pred = y_test, clf.predict(X_test)
        #print(classification_report(y_true, y_pred))
        print()
        
            

#def boostEnsemble()            

print "\n\n\n=============================  NEW RUN ===============================================\n"    
#global GLOBAL_VEC                               
#getAllTrainFeats()

FitandPredictSVM(0,9)


#FitandPredictDT()
#FitandPredictKNN()  ## try changing n_neighbors
