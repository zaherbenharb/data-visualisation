#!/usr/bin/env python
# coding: utf-8

# In[357]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# In[358]:


data=pd.read_csv("titanic-passengers.csv",sep=";")
data['Embarked'].value_counts()


# In[359]:


data.head()


# In[360]:


data.drop(['Ticket','Cabin','Fare','PassengerId','Name'],axis=1,inplace=True)


# In[361]:


data['Age'].fillna(data['Age'].mean(),inplace=True)


# In[362]:


data.head()


# In[363]:


data.describe()


# In[364]:


encoder=LabelEncoder()
data['Survived']=encoder.fit_transform(data['Survived'])
data['Embarked']=encoder.fit_transform(data['Embarked'])
one_hot=pd.get_dummies(data['Sex'])
data=data.join(one_hot)


# In[365]:


data.describe()


# In[366]:


data.dropna(axis=0,inplace=True)
data.describe()


# In[367]:


data['Pclass'].value_counts().plot.bar()


# In[368]:


data['Age'].hist()


# In[369]:


data.groupby(['Sex','Age']).mean().plot()


# In[370]:


data.groupby(['Sex']).mean().hist()


# In[371]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[372]:


plot_correlation_map(data)


# In[373]:


plt.plot(data['Sex'],data['Age'])


# In[374]:


data['Sex'].hist()


# In[385]:


sns.boxplot(x='Survived',y='Age',data=data, hue ='Sex')


# In[384]:


sns.boxplot(x='Pclass',y='Age',data=data, hue ='Sex')


# In[396]:


sns.barplot(x='Sex', y= 'Age',data=data, hue = 'Survived')


# In[ ]:




