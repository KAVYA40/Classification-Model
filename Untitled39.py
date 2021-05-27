#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
data=pd.read_excel('root.xlsx')
data.head()


# In[7]:


import re
import nltk
nltk.download('stopwords')


# In[8]:


data['Text'].dropna(inplace=True)


# In[9]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(data)):
    rev=re.sub('[^a-zA-Z]',' ',str(data['Text'][i]))
    rev=rev.lower()
    rev=rev.split()
    rev=[ps.stem(word) for word in rev if not word in stopwords.words('english')]
    rev=' '.join(rev)
    corpus.append(rev)


# In[10]:


corpus


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
x=cv.fit_transform(corpus).toarray()


# In[12]:


x.shape
y=data['Target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[13]:


cv.get_feature_names()[:30]


# In[14]:


cv.get_params()


# In[15]:


count_df=pd.DataFrame(x_train,columns=cv.get_feature_names())


# In[16]:


count_df.head()


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=7000,ngram_range=(1,3))
x=cv.fit_transform(corpus).toarray()
## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[18]:


count_df = pd.DataFrame(x_train, columns=cv.get_feature_names())
count_df.head()


# In[19]:


from sklearn import metrics
import numpy as np
import itertools
from sklearn.svm import SVC
sv_clf=SVC(probability=True,kernel='linear')
sv_clf.fit(x_test,y_test)
pred=sv_clf.predict(x_test)
score=metrics.accuracy_score(y_test,pred)
print("accuracy: %0.3f" % score) 
cm=metrics.confusion_matrix(y_test,pred)
print(cm)


# In[25]:


import matplotlib.pyplot as plt  
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
x, y = make_classification(random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(x_train, y_train)
SVC(random_state=0)
plot_confusion_matrix(clf, x_test, y_test)  
plt.show()  


# In[ ]:




