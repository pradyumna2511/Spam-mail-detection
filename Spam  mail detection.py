#!/usr/bin/env python
# coding: utf-8

# In[5]:


#SPAM MAIL DETECTION WITH ML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


raw_mail_data = pd.read_csv('spam.csv',encoding='latin-1')


# In[7]:


raw_mail_data


# In[8]:


# replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[9]:


mail_data.head()


# In[12]:


# separating the data as texts and label

X = mail_data['v2']

Y = mail_data['v1']


# In[11]:


# label spam mail as 0;  ham mail as 1;

mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1


# In[13]:


print(X)


# In[14]:


print(Y)


# Splitting the data into training data & test data

# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[16]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[17]:


# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[18]:


print(X_train)


# In[19]:


print(X_train_features)


# In[20]:


model = LogisticRegression()
model.fit(X_train_features, Y_train)


# In[21]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on test data : ', accuracy_on_test_data)


# In[22]:


#make predictions
input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]

input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




