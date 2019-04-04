# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:06:58 2018

@author: DELL
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from numpy import nan as NA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
import os
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


os.chdir("D:\Data Science\Titanic Dataset")
data = pd.read_csv('train.csv')
data.describe()
data.head()
data.isnull().any
data.dtypes
pd.crosstab(data.Sex, data.Survived)    # gives individual counts for asked variables

# Finding male v/s female ratio
y = data['Sex'].value_counts()
y
type(y)
y[0]
a = len(data['Sex'])
a
malepercent = y[0] / a
malepercent
femalepercent = y[1] / a
femalepercent

# Finding % survived

f,ax = plt.subplots(1,2,figsize=(10,6))
labels = 'Survived', 'Non-Survived'
data['Survived'].value_counts().plot.pie(explode = [0, 0.1], labels = labels, autopct = '%1.1f%%', ax = ax[0], shadow = True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data = data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()





# Plotting histogram
sarr = np.array(data['Sex'])
sarr
plt.hist(sarr, bins = 6)
plt.show()

# Find % of survival or non-survival
b = pd.crosstab(data.Sex, data.Survived)
type(b)
b = b.reset_index()
b['total'] = b.sum(axis = 1)
b
b[0] = (b[0]/ b['total']) * 100
b[1] = (b[1]/ b['total']) * 100
b

# Histogram of the whole data

data['age_bins'] = pd.cut(data['Age'], bins = [0, 25, 50, 75, 100], labels = ['<25', '25-50','50-75','75-100'])
data.head()
data.isnull().any()  # to check nan/missing values
data.isnull().sum()
type(data['age_bins'])
data['age_bins'].dropna(axis = 0, inplace = True)   # inplace overrides the original answer
data['age_bins'].value_counts()
plt.hist(data['age_bins'])
plt.show()

# Plotting stacked plot

f, ax = plt.subplots(figsize=(4, 3))
b[['Sex',0 , 1]].plot(kind='bar', stacked=True, figsize=(4,3))
plt.xticks(rotation=70 , fontsize= 12)
plt.show()

# % embarked v/s survival

d = pd.crosstab(data.Survived, data.Embarked)
d = d.reset_index() # Instead of two rows, we get 2 columns orvice versa
d
d['totalem'] = d.sum(axis = 1)
d
d[0] = (d[0] / d['totalem']) * 100
d[1] = (d[1] / d['totalem']) * 100
d

# Distribution of Prices

pricearr = np.array(data['Fare'])
pricearr
plt.hist(pricearr, bins  = 10)
data['Fare'].value_counts()
plt.show()

# Find % survival/non-survival in male v/s  female

e = pd.crosstab(data.Survived, data.Age)
e
e = e.reset_index() # Instead of two rows, we get 2 columns orvice versa
e
e['totalage'] = e.sum(axis = 1)
e
e[0] = (e[0] / e['totalage']) * 100
e[1] = (e[1] / e['totalage']) * 100
e


# Age_bin v/s Survival rate

f = pd.crosstab(data.age_bins, data.Survived)
f
f = f.reset_index()
f
f['totalsurv'] = f.sum(axis = 1)
f
f[0] = (f[0] / f['totalsurv']) * 100
f[1] = (f[1] / f['totalsurv']) * 100
f

# Parch v/s survival rates

g = pd.crosstab(data.Parch, data.Survived)
g
g = g.reset_index()
g
g['totalparch'] = g.sum(axis = 1)
g
g[0] = (g[0] / g['totalparch']) * 100
g[1] = (g[1] / g['totalparch']) * 100
g


# Binning fare & comparing with survival rate

fare = np.array(data['Fare'])
fare
bins = [5, 20, 40, 60, 85, 100, 300]
data['fare_bins'] = pd.cut(fare, bins)
data['fare_bins']
a = pd.crosstab(data.fare_bins, data.Survived)
a
a = a.reset_index()
a
a['totalfares'] = a[[0,1]].sum(axis = 1)
a
a[0] = (a[0] / a['totalfares']) * 100
a[1] = (a[1] / a['totalfares']) * 100
a

# finding sex v/s survived

data.groupby(['Sex','Survived'])['Survived'].count()
f, ax = plt.subplot(1, 2, figsize = (18,8))



























