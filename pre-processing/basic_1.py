import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics import jaccard_similarity_score
from scipy.stats import pearsonr

path = os.getcwd()+'/stemmed_output/' #path to the dataset

documents_final = []
for filename in os.listdir(path):
	file1 = open(path+filename,'r')
	text1 = file1.readlines()
	#print(text1)
	documents_final.append(str(text1))

documents = tuple(documents_final)



tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
arr = tfidf_matrix.toarray()
#for doc in arr:
	#print(doc)
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#print(cos_sim)

#print(euclidean_distances(tfidf_matrix,tfidf_matrix))

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)
count_arr = count_matrix.toarray()
total_no_of_words = count_arr.shape[1]

#Jaccard Similarity
def jaccard_similarity(query, document):
	intersection = set(query.split()).intersection(set(document.split()))
	union = set(query.split()).union(set(document.split()))
	return len(intersection)/len(union)
	
#Dice Similarity
def dice_similarity(query,document):
	intersection = set(query.split()).intersection(set(document.split()))
	return (2*len(intersection)/(len(query.split())+len(document.split())))

#Overlap Coefficient
def overlap_coefficient(query,document):
	intersection = set(query.split()).intersection(set(document.split()))
	minimum = min(len(set(query.split())),len(set(	document.split())));
	return len(intersection)/minimum

#Correlation : assumed to be Pearson correlation coefficient 
#TO DO: confirm with sir
#no simple formula associated. Cannot calculate
def correlation(query,document):
	dq = [query,document]
	count_vectorizer = CountVectorizer()
	dq_matrix = count_vectorizer.fit_transform(dq).toarray()


#Simple Matching Coefficient
def smc(query,document):
	count_vectorizer = CountVectorizer()
	doc_matrix = count_vectorizer.fit_transform(documents).toarray()
	query_matrix = count_vectorizer.fit_transform(query).toarray()
	s =0.0
	n =0.0
	for i in range(doc_matrix.shape[0]):
		n =0.0
		s =0.0
		for j in range(doc_matrix.shape[1]):
			if (query_matrix[0][j] == 0) and (doc_matrix[i][j] == 0):
				s +=1
			if  (query_matrix[0][j] == 1) and (doc_matrix[i][j] == 1):
				s +=1
			if  (query_matrix[0][j] == 1) and (doc_matrix[i][j] == 0):
				n +=1
			if  (query_matrix[0][j] == 0) and (doc_matrix[i][j] == 1):
				n +=1
		n +=s
		print(s/n)
      
for doc in documents:	
 	for doc1 in documents:	
			print(jaccard_similarity(doc,doc1),end = " ")
print("\n")
for doc in documents:	
 	for doc1 in documents:
			print(dice_similarity(doc,doc1),end = " ")
