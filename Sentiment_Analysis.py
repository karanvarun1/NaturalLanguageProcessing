import pandas as pd
import sqlite3
from nltk.corpus import stopwords
from stemming.porter2 import stem
import string
import re
import numpy as np
import math
from textblob import TextBlob
import operator

df = pd.read_csv('reviews.csv', encoding = 'utf-8')
df = (df.sort_values(by=['location_id']))
df = df.apply(lambda x: x.astype(str).str.lower())
df2 = df.groupby('location_id').apply(lambda x: x.sum())

distloc = df.location_id.unique()
print(distloc)

for i in range(len(df2)):
    regex = re.compile('[%s]' % re.escape(string.digits))
    df2.iloc[i,4] = regex.sub('',str(df2.iloc[i,4]))
#    break

for i in range(len(df2)):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    df2.iloc[i,4] = regex.sub('',str(df2.iloc[i,4]))
#    break

sw = stopwords.words("english")
for i in range(len(df2)):
    df2.iloc[i,4] = ' '.join([word for word in df2.iloc[i,4].split() if word not in sw])
#    break

for i in range(len(df2)):
    df2.iloc[i,4] = " ".join([word for word in df2.iloc[i,4].split() if (len(word)>3)])
#    break

for i in range(len(df2)):
    df2.iloc[i,4] = ' '.join(set(df2.iloc[i,4].split()))
#    break

for i in range(len(df2)):
    df2.iloc[i,4] = " ".join([stem(word) for word in df2.iloc[i,4].split()])
#    break



bloblist = []
for i in range(len(df2)):
    bloblist.append(TextBlob(df2.iloc[i,4]))
    
keywords = {}
for i, blob in enumerate(bloblist):
    tfidf = {}
    for term in blob.words:
        tf = blob.words.count(term)/len(blob.words)
        nd = len(bloblist)
        ndwt = sum(1 for b in bloblist if term in b)
        idf = math.log(nd/(1+ndwt))
        tfidf[term] = round(tf*idf, 5)
        keywords[i] = tfidf
identifier = {}

for k,v in keywords.items():
    identifier[k] =  (max(v.items(),key = operator.itemgetter(1))[0])
i = 0
words = {}
for k,v in identifier.items():
    words[distloc[i]] = v
    i = i+1
print(words)