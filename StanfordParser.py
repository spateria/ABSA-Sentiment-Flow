import xml.etree.ElementTree as ET
import nltk

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


from nltk.parse.stanford import StanfordDependencyParser
#import sys
#sys.path.append('F:/FROM_HOME_NEW/stanford-parser-full-2015-12-09')
import os
java_path = "C:/Program Files/Java/jdk1.8.0_71/bin/java.exe"
os.environ['JAVAHOME'] = java_path

_CLASS_PATH = "."    
if os.environ.get('CLASSPATH') is not None:
    _CLASS_PATH = os.environ.get('CLASSPATH')
os.environ['CLASSPATH'] = _CLASS_PATH + ';C:/Python27/Lib/slf4j-1.7.13/slf4j-log4j12-1.7.13.jar'

#jp = "C:/Program Files/Java/jdk1.8.0_71/bin/slf4j-1.7.13/slf4j-log4j12-1.7.13.jar"
#os.environ['CLASSPATH'] = jp

path_to_jar = 'D:/SEMEVAL/stanford-parser-full-2015-04-20/stanford-parser.jar'
path_to_models_jar = 'D:/SEMEVAL/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

'''

global_path = 'D:/SEMEVAL/FROM_HOME_NEW/'
################################################################
train = global_path+'ABSA/archive/train1.xml'
trial = global_path+'ABSA/archive/trial1.xml'


################################################################

def read_data(data_file):

    data = defaultdict(list)

    orig_stdout = sys.stdout
    f = file(global_path+'ABSA/parsed_sents.txt', 'w')
    sys.stdout = f
        
        
    count = 0  
    xml = ET.parse(data_file)
    root = xml.getroot()
    for sentence in root.iter('sentence'):
        count += 1
        text = sentence.find('text').text
        
        result = dependency_parser.raw_parse(text)
        dep = result.next()
        l = len(list(dep.triples())[:][:])
        for x in range(l):
            print list(dep.triples())[:][:][x]
        #if count > 10:
            #break
    
        print "\n$$$$\n"
    sys.stdout = orig_stdout
    f.close()
        
#read_data(train)'''