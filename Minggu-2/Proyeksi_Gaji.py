#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Data_Gaji.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[10]:


X = df.iloc[:, :-1]
y = df.iloc[:, 1]


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# In[12]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)


# In[13]:


y_pred = regressor.predict(X_test)


# In[17]:


import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color="green")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Tahun Pengalaman VS Gaji")
plt.xlabel("Tahun Pengalaman")
plt.ylabel("Gaji")
plt.show()


# In[18]:


plt.scatter(X_test, y_test, color="green")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Tahun Pengalaman VS Gaji")
plt.xlabel("Tahun Pengalaman")
plt.ylabel("Gaji")
plt.show()


# In[21]:


salary_pred = regressor.predict([[10]])
print("Gaji untuk tahun pengalaman itu : ", salary_pred)


# In[ ]: