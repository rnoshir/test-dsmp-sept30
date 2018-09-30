
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[100]:


df = pd.read_csv('/home/noshir/Downloads/train.csv')
df.head()


# In[101]:


correlation_values = df.select_dtypes(include=[np.number]).corr()
# corr_value[["SalePrice"]] > 0.7
# corr_value[["SalePrice"]].where(corr_value["SalePrice"] > 0.7)
selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]
selected_features


# In[120]:


X = df[["OverallQual", "TotalBsmtSF" , '1stFlrSF', 'GrLivArea','GarageCars', 'GarageArea']]


# In[121]:


X.head()


# In[122]:


y = df['SalePrice']


# In[123]:


X_train, X_test, y_train, y_test = tts(X,y, test_size=0.3, random_state = 42)


# In[124]:


reg = LinearRegression()


# In[129]:


reg.fit(X_train, y_train)


# In[130]:


y_pred = reg.predict(X_test)


# In[133]:


reg.score(X_test, y_test)


# In[134]:


rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rmse

