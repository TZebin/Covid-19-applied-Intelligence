# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:00:06 2020

@author: User
"""

import pandas as pd

import os


csvPath = os.path.sep.join(['C:/Users/User/Desktop/keras-covid-19_Adrian/covid-chestxray-dataset', "metadata.csv"])
df = pd.read_csv(csvPath)

df1 = df['view']=="PA"

df2=df[df1]
df2.to_csv('PA_covid.csv', header=True)


df3 = df2['finding']=="COVID-19"
df4=df2[df3]

df4.to_csv('covid_only.csv', header=True)


csvPath2 = os.path.sep.join(['C:/Users/User/Desktop/keras-covid-19_Adrian/covid-chestxray-dataset', "covid_only.csv"])
metadata = pd.read_csv(csvPath2)




import matplotlib.pyplot as plt
import numpy as np


print(metadata.columns)

# Sex distribution (nan if unknown)
ax = metadata['sex'].value_counts(dropna=False).plot.pie(y='sex', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Sex Distribution')


# Now age (nan in unknown)
out = pd.cut(metadata['age'], bins=np.arange(0,110,20).tolist(), include_lowest=True)
ax = out.value_counts(sort=True, dropna=False).plot.bar(rot=0, color="b", figsize=(15,6), title= "Age Distribution")
#ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
plt.show()


# Survival 
ax = metadata['survival'].value_counts(dropna=False).plot.pie(y='survival', legend = True, autopct='%2.0f%%', figsize = (5,5), title = 'Survival Distribution')

















import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio('C:/Users/User/Desktop/keras-covid-19_Adrian/dataset', output="C:/Users/User/Desktop/keras-covid-19_Adrian/dataset/output", seed=1337, ratio=(.7, .1, .2)) # default values

# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# split_folders.fixed('input_folder', output="output", seed=1337, fixed=(100, 100), oversample=False) # default values