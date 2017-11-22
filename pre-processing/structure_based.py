from stemming.porter2 import stem
import os
import string
import shutil
import re
import scipy
import math
import numpy as np
import collections
from gensim import corpora, models, similarities

path = os.getcwd()+'/query_docs/' #path to the dataset

#path to the output of stemmed data set
if os.path.exists('stemmed_output/'):
    shutil.rmtree(os.getcwd()+'/stemmed_output/')
os.mkdir(os.getcwd()+'/stemmed_output/')

import operator as op

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
d = []  # contains lists of sentences, each list of words
for filename in os.listdir(path):
    temp=""
    file1 = open(path+filename,'r')
    file2 = open(os.getcwd()+'/stemmed_output/'+filename, 'w')
    text1 = file1.readlines()
    for line in text1:
        # preprocessing the text
        line2, line_list = preprocess(line)
        file2.write(line2)
        documents.append(line_list)
        temp+=str(line2)
   # print(temp)
    d.append(temp)

#Inversions- Count
#We slightly modified the Standard algorithm to count number of Inversions 

count = 0 #the number of inversions needed for a list 'x' 
def inversionsCount(x): # time complexity is O(nlog(n))
    global count
    midsection = int(len(x) / 2)
    #print(midsection)
    leftArray = x[:midsection]
    rightArray = x[midsection:]
    if len(x) > 1:
        # Divide and conquer with recursive calls
        # to left and right arrays similar to
        # merge sort algorithm
        inversionsCount(leftArray)
        inversionsCount(rightArray)
        
        # Merge sorted sub-arrays and keep
        # count of split inversions
        i, j = 0, 0
        a = leftArray; b = rightArray
        for k in range(len(a) + len(b) + 1):
            if a[i] <= b[j]:
                x[k] = a[i]
                count += (len(b) - j)
                i += 1
                if i == len(a) and j != len(b):
                    while j != len(b):
                        k +=1
                        x[k] = b[j]
                        j += 1
                    break
            elif a[i] > b[j]:
                x[k] = b[j]
                j += 1
                if j == len(b) and i != len(a):
                    while i != len(a):
                        k+= 1
                        x[k] = a[i]
                        i += 1                    
                    break   
    return x

tlist = [] #term list of all documents
for x in d:
    y = str(x).split()
    #print(y)
    tlist.append(y)
    
#for x in tlist:
#    print(x)

#Main function to calculate Structural Similarity
ss = []
for i in range(0,len(tlist)):
    tempDict = collections.OrderedDict.fromkeys(tlist[i])
    count1 = 0;
    for j in tempDict.keys(): #indexing for document1
        tempDict[j] = count1
        count1 = count1+1
    

    for j in range(i+1,len(tlist)):
        tempDict2 = collections.OrderedDict()
        for k in range(0,len(tlist[j])):
            if tlist[j][k] in tempDict and tlist[j][k] not in tempDict2:
                #print(k)
                tempDict2[tlist[j][k]] = k
        #print(tempDict2)
        finalList = [] #gives the indexes of doc2 which are common with doc1
            finalList.append(tempDict[k])
        for k in tempDict2.keys():
       
        inversionsCount(finalList)
        numOfInv = count
        count = 0    #we reset the global variable count after every iteration
        #print("\nNo. of Inversions: "+str(numOfInv))
        if len(finalList) <= 1:
            #If there are less than 1 common word,
            #we take the similarity to be '0'
            ss.append(0)
        else:
            #otherwise, we calculate the value of a/b
            ss.append("{:2.4f}".format((2*numOfInv)/(len(finalList)*(len(finalList)-1))))

print(len(ss))
print(ss)   
