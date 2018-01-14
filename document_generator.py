import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from collections import Counter
import numpy as np
import numpy.linalg as LA
import operator

#Functions
#cosine similarity
cosine_function = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

#***NEED TO CONDITION TO STEM ONLY NON PROPER NOUNS***
#***NEED TO REMOVE PUNCTUATIONS BEFORE TOKENIZING***
#Pre-process: generate word frequency dictionary for a given sentence
def sen_to_dict(sen):
    words_punct = word_tokenize(sen)
    #remove punctuations
    just_words = [i for i in words_punct if i not in string.punctuation]
    stop_words = stopwords.words('english')
    main_words = [i for i in just_words if i not in stop_words]
    stemmer = PorterStemmer()
    stemmed_main_words = []
    for w in main_words:
        #NEED TO CONDITION TO STEM ONLY NON PROPER NOUNS
        stemmed_main_words.append(stemmer.stem(w))
    
    sen_dict = dict(Counter(stemmed_main_words))
    return sen_dict

#get the given document representation as a list of sentences; where each sentence is represented as word frequency dictionary
def get_sen_doc(doc):
    sentences = sent_tokenize(doc)
    sen_doc = [sen_to_dict(x) for x in sentences]
    return sen_doc

#get all the unique words in the document
def list_of_words(list):
    wordDict = Counter()
    for x in list:
        wordDict = wordDict + Counter(x) 

    return wordDict

#normalizing each sentence
def normalize_tf(list):
    for k in list:
        length = float(len(k))
        for key, value in k.iteritems():
            value = value/length
            k[key] = value
    return list

#calculate isf
def calc_isf(list,wordDict):
    n = len(wordDict)
    m = len(list)
    isf = {}
    isf = isf.fromkeys(wordDict.keys(),0)
    #calculate number of sentences with term t
    for k in list:
        for key, value in isf.iteritems():
            if k.has_key(key):
                isf[key] = isf[key] + 1
    #calculate isf
    for key, value in isf.iteritems():
        isf[key] = np.log(m) - np.log(isf[key])
    return isf

#filling the tsisf dict
def fill_tsisf(list, wordDict,isf):
    n = len(wordDict)
    m = len(list)
    tsisf = []
    dict = {}
    for k in list:
        dict = dict.fromkeys(wordDict.keys(),0)
        dict2 = {x: (dict.get(x, 0) + k.get(x, 0))*(isf.get(x,0)) for x in set(dict).union(k)}
        tsisf.append(dict2)
        dict.clear()
    return tsisf

#convert tsisf dict to array
def get_tsisf(list,wordDict,tsisf):
    n = len(wordDict)
    m = len(list)
    b = np.zeros((m,n))
    c = 0
    for i in list:
        j = 0
        for key, value in sorted(wordDict.items(), key=lambda x: x[1]):
            b[c][j] = tsisf[c][key]
            j = j + 1
        c = c + 1
    return b

#calculate whole document's tsisf
def doc_tsisf(list,wordDict,tsisf):
    n = len(wordDict)
    m = len(list)
    doc = np.zeros(n)
    for i in range(n):
        for j in range(m):
            doc[i] = doc[i] + tsisf[j][i]
    return doc

#get cosine similarity for each sentence
def get_cossim(list,wordDict,tsisf,doc):
    n = len(wordDict)
    m = len(list)
    cossim = np.zeros((m,1))
    for i in range(m):
        cossim[i] = cossim[i] + cosine_function(tsisf[i], doc)
    return cossim

#BaselineFeature (B1): Get cosine similarity from sent_document dict
def get_doc_cosine_dist(doc):
    sen_doc = get_sen_doc(doc)
    wordDict = list_of_words(sen_doc)
    normalize_tf(sen_doc)
    isf = calc_isf(sen_doc,wordDict)
    tsisf = fill_tsisf(sen_doc,wordDict,isf)
    arr = get_tsisf(sen_doc,wordDict,tsisf)
    doc2 =  doc_tsisf(sen_doc,wordDict,arr)
    return get_cossim(sen_doc,wordDict,arr,doc2)

#BaselineFeature (B2): Retrieves the real valued position of each sentence in the document
def get_real_valued_pos(doc):
    sen_doc = get_sen_doc(doc)
    m = len(sen_doc)
    real_valued_pos = np.zeros((m,1))
    for i in range(m):
        real_valued_pos[i] = float(i)/float(m)
    return real_valued_pos

#BaselineFeature (B345): Retrieves *3* binary features describing the first third, second third and third third respectively
def get_n_third_features(doc):
    real_valued_pos = get_real_valued_pos(doc)
    m = len(real_valued_pos)
    n_thrid_features = np.zeros((m,3))
    first_third = np.array([1,0,0])
    second_third = np.array([0,1,0])
    third_third = np.array([0,0,1])
    for i in range(m):
        if real_valued_pos[i][0] < 0.333334:
            n_thrid_features[i] = first_third
        elif real_valued_pos[i][0] < 0.666667:
            n_thrid_features[i] = second_third
        else:
            n_thrid_features[i] = third_third
    return n_thrid_features
    
#BaselineFeature (B6): Retrieves a binary feature whether the sentence is of medium length; Medium Length = 4-15
def get_medium_length(doc):
    sen_doc = get_sen_doc(doc)
    m = len(sen_doc)
    medium_len_feature = np.zeros((m,1))
    for i in range(m):
        sen_len = len(sen_doc[i])
        if sen_len >=4 and sen_len <= 15:
            medium_len_feature[i] = 1
        else:
            medium_len_feature[i] = 0
    return medium_len_feature

#Aggregating all features
def aggregate_features(doc):
    b1 = get_doc_cosine_dist(doc)
    b2 = get_real_valued_pos(doc)
    b345 = get_n_third_features(doc)
    b6 = get_medium_length(doc)
    baseline_feature_vector = np.hstack((b1,b2,b345,b6))
    return baseline_feature_vector

def get_summary(doc, num_of_sent):
    arr = get_doc_cosine_dist(doc)
    x = dict(enumerate(arr))
    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    sentences = sent_tokenize(test_doc)
    sent_num = []
    for i in sorted_x:
        sent_num.append(i[0])
    num = sent_num[:num_of_sent]
    num.sort()
    summary = "".join([sentences[x].strip() for x in num])
    return summary


#Main
#example document
test_doc_fp = open('/home/vinay/temp.txt','r')
test_doc = test_doc_fp.read()
#p = aggregate_features(test_doc)
#print p
print get_summary(test_doc, 10)
test_doc_fp.close()
#End
