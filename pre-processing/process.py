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
#np.seterr(divide='ignore', invalid='ignore')

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
i=0
for filename in os.listdir(path):
	file1 = open(path+filename,'r')
	if(i==296):
		i+=1
		continue		
	text1 = file1.readlines()
	
	documents_final.append(str(text1))
	i+=1

documents = tuple(documents_final)
#print(documents_final[296])	
############-------pre-preprocessing ends here-------###########
############-------calculation of similarity coefficients---####
#cosine_similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
tf_arr = tfidf_matrix.toarray()

cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#euclidean_distances
euc_dis = euclidean_distances(tfidf_matrix,tfidf_matrix)

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
	p =[]
	p = pearsonr(query,document)
	return (p[0]+p[1])/2


#Simple Matching Coefficient
def smc(query,document):
	dot = np.dot(query,document,None)
	q_len = np.linalg.norm(query)
	d_len = np.linalg.norm(document)
	return dot/(q_len + d_len)

matrix1 = [[0 for i in range(297)] for j in range(297)]
matrix2 = [[0 for i in range(297)] for j in range(297)]
matrix3 = [[0 for i in range(297)] for j in range(297)]
matrix4 = [[0 for i in range(297)] for j in range(297)]
matrix5 = [[0 for i in range(297)] for j in range(297)]

for i in range(tf_arr.shape[0]):
	for j in range(tf_arr.shape[0]):
		matrix1[i][j] = jaccard_similarity(tf_arr[i],tf_arr[j])
		matrix2[i][j] = dice_similarity(tf_arr[i],tf_arr[j])
		matrix3[i][j] = overlap_coefficient(tf_arr[i],tf_arr[j])
		matrix4[i][j] = smc(tf_arr[i],tf_arr[j])
		matrix5[i][j] = correlation(tf_arr[i],tf_arr[j])

####-------calculation of similarity coefficients ends here---####
final_matrix = {}
class_matrix = {}
n = len(documents) #298
print(n)
mat = []
for i in range(n):
	d = (documents[i])
	index1 = d.find(" ")
	#print(index1)
	index2 = d[index1+1:].find(" ")
	#print(index2)
	name1 = documents[i][0:index1]	+ '-' + documents[i][index1+1:index1+index2+1]
	name1 = name1[2:].upper()	
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
		new_list = []
		new_list.append(cos_sim[i][j])
		new_list.append(euc_dis[i][j])
		new_list.append(matrix1[i][j])
		new_list.append(matrix2[i][j])
		new_list.append(matrix3[i][j])
		new_list.append(matrix4[i][j])
		new_list.append(matrix5[i][j])
		if(path_list[name1]==path_list[name2]):
			class_matrix[name1 + '*' + name2]= 1
			mat.append(1)
		else:
			class_matrix[name1 + '*' + name2]= 0
			mat.append(0)


		final_matrix[name1 + '*' + name2]= new_list

print(final_matrix["AP900107-0033*AP881104-0268"])
print(final_matrix["AP900107-0033*WSJ871022-0085"])
#print(type(list(final_matrix.values())))
#print(final_matrix.values().shape[0])
#print(len(final_matrix)) #44253

from sklearn import svm, datasets

C = 1.0 
X= list(final_matrix.values())
y = mat
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
print(svc.predict([[0.069895674483874626, 1.3638946627332511, 0.036213417875836255, 0.069895674483874543, 0.069895674483874556, 0.034947837241937264, 0.029920721004898222]]))

'''[0.54163853695496034, 0.95745648783121151, 0.37140211852830096, 0.54163853695496023, 0.54163853695496034, 0.27081926847748011, 0.26864157307494591]
[0.069895674483874626, 1.3638946627332511, 0.036213417875836255, 0.069895674483874543, 0.069895674483874556, 0.034947837241937264, 0.029920721004898222]
[1]'''

# p=0
# length = 0
# l1=[]

# for i in range(298):
#     d1 = documents_final[i].split()
#     l2=[]
#     for j in range(298):
#         if(j<=i):
#             l2.append(0)	
#         else:
#             d2= documents_final[j].split()
#             doc=set(d1)&set(d2)
#             doc1=sorted(doc,key = lambda k :d1.index(k))
#             doc2=sorted(doc,key = lambda k :d2.index(k))
#             length = len(doc1)
#             #print(length)
#             #print("\n")
#             if(length>1):
#                 n=0
#                 for i in range(length):
#                     n += len(set(doc1[i:])&set(doc2[doc2.index(doc1[i]):]))
#                 #print(n)
               
#                 p = 2*n/(length*(length-1))
#                 #print(p)
#                 l2.append(p)
#             else:
#                 p = 0
#                 l2.append(p)
#     #print(l2)
#     #print("\n")
#     l1.append(l2)
# #print(l1)
