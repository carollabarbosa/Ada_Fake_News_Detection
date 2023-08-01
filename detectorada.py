#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install pandas')


# In[6]:


get_ipython().system('pip install numpy')


# In[8]:


import pandas as pd
import numpy as np
import itertools


# In[9]:


df = pd.read_csv("FakeTrue21.csv")


# In[10]:


df.head()


# In[11]:


df.shape


# In[12]:


df.isnull().sum()


# In[13]:


labels = df.label


# In[14]:


labels.head()


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(df["text"], labels, test_size = 0.2, random_state = 20)


# In[17]:


x_train.head()


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


# In[19]:


# initilise a Tfidvectorizer
vector = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[20]:


# fit and tranform
tf_train = vector.fit_transform(x_train)
tf_test = vector.transform(x_test)


# In[21]:


# initilise a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train, y_train)


# In[22]:


# predicton the tst dataset
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = pac.predict(tf_test)


# In[23]:


score = accuracy_score(y_test, y_pred)


# In[24]:


print(f"Accuracy : {round(score*100,2)}%")


# In[25]:


# confusion metrics
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])


# In[26]:


# save model
import pickle
filename = 'finalized_model.pkl'
pickle.dump(pac, open(filename, 'wb'))


# In[27]:


# save vectorizer
filename = 'vectorizer.pkl'
pickle.dump(vector, open(filename, 'wb'))


# In[ ]:




