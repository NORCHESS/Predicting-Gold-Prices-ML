#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Plotting the correlation matrix of all our variables
# Importing libraries  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sns


# In[2]:


# Importing dataset, filling in the missing values and Illustrating it
data = pd.read_excel('data.xlsx', sheet_name='All Variables')
data = data.fillna(method="ffill",axis=0)
data = data.fillna(method="bfill",axis=0)


# In[3]:


# Correlation Matrix
corr = data.corr(method = 'pearson')   
plt.figure(figsize= (8, 8))
sns.heatmap(corr, cbar=True, square=True, fmt = '.2f', annot=True, annot_kws={'size': 7.8}, cmap = 'Blues')
plt.savefig('correlation_matrix.png',  transparent=True, dpi=200)


# In[ ]:




