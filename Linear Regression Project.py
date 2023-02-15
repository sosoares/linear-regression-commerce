#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('Ecommerce Customers')


# In[5]:


df.columns


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


sns.jointplot(x= df['Time on Website'], y= df['Yearly Amount Spent'])


# In[9]:


sns.jointplot(x= df['Time on App'], y=df['Yearly Amount Spent'])


# In[10]:


sns.jointplot( x=df['Time on App'], y= df['Length of Membership'], kind= 'hex')


# In[11]:


df = df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership', 'Yearly Amount Spent']]


# In[12]:


sns.pairplot(df)


# In[13]:


sns.lmplot(x= 'Yearly Amount Spent', y= 'Length of Membership', data= df)


# In[14]:


df.columns


# In[15]:


X = df[['Avg. Session Length', 'Time on App', 'Time on Website',
       'Length of Membership']]


# In[16]:


y = df['Yearly Amount Spent']


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


lm = LinearRegression()


# In[21]:


lm.fit(X_train, y_train)


# In[22]:


print(lm.intercept_)


# In[23]:


lm.coef_


# In[24]:


xd = pd.DataFrame (lm.coef_, X.columns, columns = ['Coeficiente'])
xd


# In[25]:


predictions = lm.predict(X_test)


# In[26]:


sns.scatterplot(x = y_test, y= predictions)


# In[27]:


from sklearn import metrics


# In[28]:


metrics.mean_absolute_error(y_test, predictions)


# In[29]:


metrics.mean_squared_error(y_test, predictions)


# In[30]:


np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[31]:


sns.histplot(y_test-predictions, bins= 50, kde = True)


# In[32]:


xd


# In[ ]:




