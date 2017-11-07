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
	documents_final.append(str(text1))

documents = tuple(documents_final)
print(type(documents[0]))