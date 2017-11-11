import os
import numpy as np
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics import jaccard_similarity_score
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
path = os.getcwd()+'/stemmed_output/' #path to the dataset

documents_final = []
for filename in os.listdir(path):
	file1 = open(path+filename,'r')
	text1 = file1.readlines()
	#print(text1)
	documents_final.append(str(text1))

documents = tuple(documents_final)
#print(documents)

#cosine_similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tf_arr = tfidf_matrix.toarray()
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Cosine Similarity:")
print(cos_sim)
print('\n')

#euclidean_distances
print("Euclidean Distances:")
euc_dis = euclidean_distances(tfidf_matrix,tfidf_matrix)
print(euc_dis)
print('\n')

count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)
count_arr = count_matrix.toarray()
total_no_of_words = count_arr.shape[1]

#Jaccard Similarity
def jaccard_similarity(query, document):
	dot = np.dot(query,document,None)
	q_len = np.linalg.norm(query)
	d_len = np.linalg.norm(document)
	q_sqr = q_len**2
	d_sqr = d_len**2
	j_s = dot/(q_sqr + d_sqr-dot)
	return j_s
	
#Dice Similarity
def dice_similarity(query,document):
	dot = np.dot(query,document,None)
	q_len = np.linalg.norm(query)
	d_len = np.linalg.norm(document)
	q_sqr = q_len**2
	d_sqr = d_len**2
	return (2*dot)/(d_sqr +q_sqr)



#Overlap Coefficient
def overlap_coefficient(query,document):
	dot = np.dot(query,document,None)
	q_len = np.linalg.norm(query)
	d_len = np.linalg.norm(document)
	minimum = min(q_len,d_len)
	return dot/minimum

def correlation(query,document):
	return pearsonr(query,document)

#Simple Matching Coefficient
def smc(query,document):
	dot = np.dot(query,document,None)
	q_len = np.linalg.norm(query)
	d_len = np.linalg.norm(document)
	return dot/(q_len + d_len)


matrix1 = [[0 for i in range(20)] for j in range(20)]
matrix2 = [[0 for i in range(20)] for j in range(20)]
matrix3 = [[0 for i in range(20)] for j in range(20)]
matrix4 = [[0 for i in range(20)] for j in range(20)]
matrix5 = [[0 for i in range(20)] for j in range(20)]

for i in range(tf_arr.shape[0]):
	for j in range(tf_arr.shape[0]):
		matrix1[i][j] = jaccard_similarity(tf_arr[i],tf_arr[j])
		matrix2[i][j] = dice_similarity(tf_arr[i],tf_arr[j])
		matrix3[i][j] = overlap_coefficient(tf_arr[i],tf_arr[j])
		matrix4[i][j] = smc(tf_arr[i],tf_arr[j])
		matrix5[i][j] = correlation(tf_arr[i],tf_arr[j])

print("\n")
print("Jaccard Similarity:")
print(matrix1)	

print("\n")
print("Dice Similarity:")
print(matrix2)	

print("\n")
print("Overlap_coefficient:")
print(matrix3)

print("\n")
print("SMC:")
print(matrix4)

print("\n")
print("Correlation:")
print(matrix5)

matrix6 = [[]]
matrix7 = [[]]
matrix6 = cos_sim
matrix7 = euc_dis
