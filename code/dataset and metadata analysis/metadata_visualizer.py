#!/usr/bin/env python
# coding: utf-8

# # Understand the type of data contained

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import data
metadata_dir = 'C:/Users/User/Desktop/keras-covid-19_Adrian/covid-chestxray-dataset/PA_covid.csv'
metadata = pd.read_csv(metadata_dir)
# what info do we have?
print(metadata.columns)


# In[2]:


#print(metadata.head(10))


# ## Plot some readily available values ( ***nan*** where data is unknown)

# In[3]:


# Finding distribution (nan if unknown)
ax = metadata['finding'].value_counts(dropna=False).plot.pie(y='Finding', legend = True, autopct='%2.0f%%', figsize = (10,10), title = 'Finding Distribution')


# In[4]:


# Sex distribution (nan if unknown)
ax = metadata['sex'].value_counts(dropna=False).plot.pie(y='sex', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Sex Distribution')


# In[5]:


# Now age (nan in unknown)
out = pd.cut(metadata['age'], bins=np.arange(0,110,10).tolist(), include_lowest=False)
ax = out.value_counts(sort=False, dropna=False).plot.bar(rot=0, color="b", figsize=(15,6), title= "Age Distribution")
#ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
plt.show()


# In[6]:


# Survival 
ax = metadata['survival'].value_counts(dropna=False).plot.pie(y='survival', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Survival Distribution')


# In[7]:


# View
ax = metadata['view'].value_counts(dropna=False).plot.pie(y='view', legend = True, autopct='%2.0f%%', figsize = (8,8), title = 'View distribution')


# In[8]:


# Modality
ax = metadata['modality'].value_counts(dropna=False).plot.pie(y='modality', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Modality Distribution')


# In[9]:


# Location
metadata['country'] = metadata['location'].apply(lambda x: x.split(',')[-1].replace(" ","") if x is not np.nan else "Unknown")
ax = metadata['country'].value_counts(dropna=False).plot.pie(y='country', legend = True, autopct='%2.0f%%', figsize = (10,10), title = 'Location of the patient')


# In[ ]:





