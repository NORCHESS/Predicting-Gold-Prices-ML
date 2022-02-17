#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Plotting the historical development of the gold price 
# Importing libraries  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from datetime import datetime
from datetime import datetime, timedelta
import seaborn as sns


# In[9]:


# Importing dataset, filling in the missing values and Illustrating it
data = pd.read_excel('data.xlsx', sheet_name='All Variables')
data = data.fillna(method="ffill",axis=0)
data = data.fillna(method="bfill",axis=0)


# In[10]:


# The Historical development of the gold price (2010-2021)
data_illustration = pd.read_excel('data.xlsx', sheet_name='Gold Price')
plt.figure(figsize=(12,7))
plt.plot('Date', 'Gold', data=data_illustration)
plt.grid(axis='y')
plt.ylabel('Price (USD)\n', size=14)
plt.title('Gold Price', size = 17)
plt.savefig('Gold Price.png', dpi=200, transparent=True)

