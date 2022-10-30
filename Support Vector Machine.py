#!/usr/bin/env python
# coding: utf-8

# # Support Vector machine #Classification & Regression

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv')


# In[3]:


df.head()


# # EDA 

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[20]:


df.corr()


# In[8]:


df.describe().T


# In[9]:


numerical_feature=[feature for feature in df.columns if df[feature].dtype!='Object']
print('The numerical feature is {} and featue are {}'.format(len(numerical_feature),numerical_feature))


# # Univariate anaylsis

# In[11]:


for feature in numerical_feature:
    sns.histplot(data=df,x=feature,kde=True,color='g')
    plt.show()


# # Observation :
# 
# # 1. Fixed acidity,Volatile acidity,citric acid, Residual sugar,chlorides, free sulfer dioxide, total sulfur dioxide, Sulphates, alcohol all feature are right sekwed.
# # 2. Ph, density normally distributed.

# In[14]:


for feature in numerical_feature:
    sns.boxplot(data=df,x=feature,color='g')
    plt.show()


# # Observation:
# 
# # 1. Fixed acidity,volatile acidity, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, sulphates, alcohol have outlier in right side.
# 
# # 2. Ph, density & chlorides have outlier in both side.

# In[18]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# # Data Spliting 

# In[22]:


X=df.drop(['quality'],axis=1)


# In[24]:


X.head()


# In[25]:


y=df['quality']


# In[26]:


y


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=10)


# In[29]:


X_train.shape


# In[30]:


X_test.shape


# In[31]:


y_train.shape


# In[32]:


y_test.shape


# # # Perform standardization

# In[33]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()


# In[34]:


X_train=scaler.fit_transform(X_train)


# In[35]:


X_test=scaler.transform(X_test)


# In[36]:


X_train


# In[37]:


X_test


# In[40]:


from sklearn.svm import SVC
model1=SVC()


# In[41]:


model1.fit(X_train,y_train)


# In[42]:


pred1=model1.predict(X_test)


# In[43]:


pred1


# In[46]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[45]:


accuracy_score(y_test,pred1)


# In[47]:


confusion_matrix(y_test,pred1)


# # SVC: accuracy socre is 61

# # Let's apply Logistic Regression 

# In[48]:


from sklearn.linear_model import LogisticRegression

logistic=LogisticRegression()


# In[49]:


logistic.fit(X_train,y_train)


# In[50]:


pred2=logistic.predict(X_test)


# In[51]:


pred2


# In[52]:


accuracy_score(y_test,pred2)


# # Observation: SVC is worked well instead of Logistic regression.

# # Let's Apply SVR

# In[54]:


from sklearn.svm import SVR

model2=SVR()


# In[55]:


model2.fit(X_train,y_train)


# In[56]:


pred3=model2.predict(X_test)


# In[57]:


from sklearn.metrics import r2_score


# In[58]:


r2_score(y_test,pred3)


# # observation: Accuracy score further down in SVR

# In[ ]:




