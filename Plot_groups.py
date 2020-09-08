from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import numpy as np
import csv
import re

documents=[]
with open('data.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                row[1]
            except:
                continue
            documents=documents+[[re.sub("[^\w']", " ",  row[1]).lower()]]
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 1000,min_df = 100)
matrix = vectorizer.fit_transform([x[0] for x in documents]).transpose()

words_compressed, _, docs_compressed = svds(matrix, k=20)
docs_compressed = docs_compressed.transpose()
word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}
words_compressed = normalize(words_compressed, axis = 1)

l=[]
for f,s in word_to_index.items():
    l.append([s,f])
print(str(sorted(l,reverse=True)))
    
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1)
#print(docs_compressed.shape)

projected_docs = tsne.fit_transform(docs_compressed)
#print(projected_docs.shape)

plt.figure(figsize=(15,15))
plt.scatter(projected_docs[:,0],projected_docs[:,1])
plt.show()

