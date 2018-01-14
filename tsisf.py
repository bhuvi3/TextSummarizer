from collections import Counter
import numpy as np
import numpy.linalg as LA

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

#cosine similarity
cosine_function = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)

#get cosine similarity for each sentence
def get_cossim(list,wordDict,tsisf,doc):
	n = len(wordDict)
	m = len(list)
	cossim = np.zeros((m,1))
	for i in range(m):
		cossim[i] = cossim[i] + cosine_function(tsisf[i], doc)
	return cossim

#get cosine similarity from sent_document dict
def get_doc_cosine_dist(doc):
	sen_doc = get_sen_doc(doc)
	wordDict = list_of_words(sen_doc)
	normalize_tf(sen_doc)
	isf = calc_isf(sen_doc,wordDict)
	tsisf = fill_tsisf(sen_doc,wordDict,isf)
	arr = get_tsisf(sen_doc,wordDict,tsisf)
	doc_tsisf =  doc_tsisf(sen_doc,wordDict,arr)
	return get_cossim(sen_doc,wordDict,arr,doc_tsisf)

#example document
list = [{'a':1, 'b':2, 'c':3},{'b':3, 'c':4, 'd':5},{'d':6, 'a':1}]

print get_doc_cosine_dist(list)
