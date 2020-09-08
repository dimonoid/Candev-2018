# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:07:36 2018

@author: ShinrayL
"""

#
#from bs4 import BeautifulSoup 
#import urllib.request 
#response = urllib.request.urlopen('file:///C:/Users/ShinrayL/Desktop/CANDEV/CANDEV.mht') 
#html = response.read() 
#soup = BeautifulSoup(html,"html5lib") 
#text = soup.get_text(strip=True) 
#tokens = [t for t in text.split()] 
#print (tokens)


from bs4 import BeautifulSoup
import urllib.request
import nltk 
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.collocations
response = urllib.request.urlopen('file:///C:/Users/ShinrayL/Desktop/CANDEV/CANDEV.mht')
html = response.read()
soup = BeautifulSoup(html,"html5lib")
text = soup.get_text(strip=True)
tekent=sent_tokenize(text)
tokens = text.split()
print (pos_tag(tokens))
clean_tokens = list()
sr = stopwords.words('english')
for token in tokens:
    if not token in sr:
        clean_tokens.append(token)
freq = nltk.FreqDist(clean_tokens)
for key,val in freq.items():
    print (str(key) + ':' + str(val))
freq.plot(30,cumulative=False)
obj =TfidfVectorizer()
X=obj.fit_transform(tekent)
most_common=freq.most_common(40)
print(most_common)