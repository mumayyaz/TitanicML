#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('D:\MachineLearning\mydata')
import pandas as pd
import numpy as np
import cv2
import sklearn.preprocessing as pre
import sklearn.linear_model as lin
import sklearn.metrics as met
import sklearn.model_selection as mod
import sklearn.feature_extraction.text as tex


# In[132]:


train = pd.read_csv('train.csv' , encoding='latin')
test = pd.read_csv('test.csv' , encoding='latin')


# In[177]:


train.isnull().sum()


# In[158]:


train.shape


# In[174]:


train.Age.value_counts().sort_values(ascending=False).head(10)


# In[200]:


train.sort_values('Age',ascending=False).head(4)


# In[178]:


train = train.fillna(value={'Age': 30 })
test = test.fillna(value={'Age': 30 })


# In[185]:


xtrain = train.drop(['PassengerId','Survived','Name','Ticket', 'Fare', 'Cabin'],axis=1)


# In[187]:


xtest = test.drop(['PassengerId','Name','Ticket', 'Fare', 'Cabin'],axis=1)


# In[188]:


ytrain = train['Survived']


# In[209]:


xtrain = pd.get_dummies(xtrain)
xtest = pd.get_dummies(xtest)


# In[211]:


from sklearn.preprocessing import StandardScaler


# In[212]:


sc = StandardScaler()


# In[214]:


xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# In[216]:


logi=lin.LogisticRegression()


# In[217]:


tit=logi.fit(xtrain,ytrain)


# In[226]:


predictions = tit.predict(xtest)


# In[228]:


submission = pd.DataFrame ({'PassengerId':test.PassengerId,'Survived':predictions})


# In[230]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




