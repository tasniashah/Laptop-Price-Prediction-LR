#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


dataset = pd.read_csv("D:/laptopPrice.csv")


# In[23]:


dataset.head()


# In[24]:


dataset.info()


# In[25]:


dataset.describe()


# In[26]:


dataset.nunique()


# In[27]:


dataset.isna().sum()


# In[28]:


dataset.shape


# In[29]:


sns.countplot(data=dataset, x='brand', order=dataset['brand'].value_counts().index)
plt.title('Number of reviews for each brand')


# In[30]:


fig, axis = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axis[0], x=dataset.processor_name)
axis[0].set_title("Count of laptops by Processor Name")
axis[0].tick_params(axis='x', rotation=90)

sns.boxplot(ax=axis[1], x=dataset.processor_name, y=dataset.Price)
axis[1].set_title("Boxplot of Prices by Processor Name")
axis[1].tick_params(axis='x', rotation=90);


# In[31]:


fig, axis = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axis[0], x=dataset.os, hue=dataset.os_bit)
axis[0].set_title("Count of laptops by OS")

sns.boxplot(ax=axis[1], x=dataset.os, y=dataset.Price, hue=dataset.os_bit)
axis[1].set_title("Boxplot of Prices by OS");


# In[32]:


dataset['processor_gnrtn'].replace('Not Available', dataset['processor_gnrtn'].mode()[0], inplace=True)


# In[33]:


# function to identify and remove outliers from the specified column of the dataframe

def removeOutlier(df, column):
    
    # sort values in ascending order in order to calculate the quartiles and the IQR 
    sorts = df[column].sort_values()
    
    Q1 = sorts.quantile(0.25)    #  25th percentile (first quartile)
    Q3 = sorts.quantile(0.75)    #  75th percentile (third quartile)
    IQR = Q3 - Q1
    
    lower_bound = sorts < (Q1-1.5*IQR)
    upper_bound = sorts > (Q3+1.5*IQR)
    # exlude rows from the dataframe above and below bounds
    df_final = df[~((lower_bound) | (upper_bound))].reset_index(drop=True)    
    
    return df_final
# apply the function to all 3 numerical columns
laptops1 = removeOutlier(dataset, "Price")
laptops2 = removeOutlier(laptops1, "Number of Reviews")
laptops_clean = removeOutlier(laptops2, "Number of Ratings")
laptops_clean.shape


# In[34]:


laptops_clean['rating'] = laptops_clean['rating'].str.split().str[0].astype('int') #splits the rating column and stores 1st part of the string as an integer
columns_to_encode = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 
                     'hdd', 'os', 'os_bit', 'graphic_card_gb', 'weight', 'warranty', 'Touchscreen', 'msoffice']
laptops_final = pd.get_dummies(laptops_clean, columns=columns_to_encode, drop_first=True)
laptops_final


# In[145]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x = laptops_final.drop(['Price'],axis=1)
y = laptops_final['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=99)


# In[146]:


model = LinearRegression()


# In[147]:


model.fit(X=x_train, y=y_train)


# In[148]:


model.coef_


# In[149]:


predictions = model.predict(x_test)


# In[150]:


plt.scatter(y_test,predictions,s=15)
plt.xlabel('Y Test(True Values)')
plt.ylabel('Predicted Values')
plt.plot(y_test, y_test, color='red', lw=1)

plt.show()


# In[151]:


from sklearn import metrics
print('Mean Absolute Error = :', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error = :', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error = :', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[153]:


print('Variance = ', metrics.explained_variance_score(y_test, predictions))


# In[154]:


sns.displot((y_test-predictions),kde=True, bins=50);

