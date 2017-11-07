from stemming.porter2 import stem
import os
import string
import shutil
import re
import numpy as np
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics import jaccard_similarity_score
from scipy.stats import pearsonr

'''list_docs = ["d01a","d09b","d18c","d25e","d35f","d42g",
"d49i","d58k","d02a","d10b","d20d","d26e","d36f","d46h",
"d51i","d60k","d03a","d16c","d21d","d29e","d38g","d47h",
"d52i","d07b","d17c","d23d","d33f","d40g","d48h","d55k"]
'''
path = os.getcwd()+'/query_docs/' #path to the dataset

#path to the output of stemmed data set
if os.path.exists('stemmed_output/'):
	shutil.rmtree(os.getcwd()+'/stemmed_output/')
os.mkdir(os.getcwd()+'/stemmed_output/')


documents = [] # contains lists of sentences, each list of words

stoplist = [] # contains the list of stopwords
f=open('stopword.txt')
for line in f:
    line = line[:-2] # the line is like "the\r\n"
    stoplist.append(line)
f.close()


def preprocess(line):
	line = line.strip()
	line = line.lower()
	line=re.sub("<.*?>","",line)
	for c in string.punctuation:
		line=line.replace(c,' ')
	line2=''  # contains the stemmed sentence 
	line_list = []
	for word in line.split():
		if word in stoplist:
			continue
		if len(word) < 3:
			continue
		stemmed_word = stem(word)
		line2+=stemmed_word+' '
		line_list.append(stemmed_word)
	return line2, line_list

# looping over the files
path_list={}
for root, dirs, files in os.walk(path):
#for filename in os.listdir(path):
	for file in files:
		file1 = open(os.path.join(root, file),'r')
		file2 = open(os.getcwd()+'/stemmed_output/'+file, 'w')
		text1 = file1.readlines()
		r = root
		ind = r.rfind("/")
		r = r[:ind]
		ind2 = r.rfind('/')
		r = r[ind2+1:]
		#print(file)
		path_list[file] = r
		for line in text1:
			# preprocessing the text
			line2, line_list = preprocess(line)
			file2.write(line2)
			documents.append(line_list)

#list to tuple		
path = os.getcwd()+'/stemmed_output/' #path to the dataset

documents_final = []
for filename in os.listdir(path):
	file1 = open(path+filename,'r')
	text1 = file1.readlines()
	#print(text1)
	documents_final.append(str(text1))

documents = tuple(documents_final)
	
#cosine_similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
arr = tfidf_matrix.toarray()

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

final_matrix = {}
n = len(documents) #303
print(n)

for i in range(n):
	d = (documents[i])
	index1 = d.find(" ")
	#print(index1)
	index2 = d[index1+1:].find(" ")
	#print(index2)
	name1 = documents[i][0:index1]	+ '-' + documents[i][index1+1:index1+index2+1]
	name1 = name1[2:].upper()
	#print(name1)	
	for j in range(i):			
		#computation
		d = (documents[j])
		index1 = d.find(" ")
		#print(index1)
		index2 = d[index1+1:].find(" ")
		#print(index2)
		name2 = documents[j][0:index1]	+ '-' + documents[j][index1+1:index1+index2+1]
		name2 = name2[2:].upper()
		#print(name2)

		final_matrix[name1 + '*' + name2]= cos_sim[i][j]

print(final_matrix)
print(len(final_matrix)) #44832
