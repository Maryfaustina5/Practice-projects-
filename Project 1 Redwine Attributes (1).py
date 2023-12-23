#!/usr/bin/env python
# coding: utf-8

# FLIPROBO PROEJECT 1

# RED WINE PROJECT

# In[5]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[6]:


# extacting the dataset from the provided link
url = "https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Red%20Wine/winequality-red.csv"
df = pd.read_csv(url)
df = pd.read_csv(url, encoding='utf-8', na_values='?')


# In[ ]:


# encoding
cutoff = 7
df['quality_binary'] = (df['quality'] >= cutoff).astype(int)


# In[ ]:


#encoding the variables
X = df.iloc[:, 0:11] # Input variables
y = df['quality_binary'] # Output variable


# In[7]:


# extracting the DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[ ]:


# code written here to make sure that I don't miss on this one.



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Red%20Wine/winequality-red.csv"
df = pd.read_csv(url)
df = pd.read_csv(url, encoding='utf-8', na_values='?')


cutoff = 7
df['quality_binary'] = (df['quality'] >= cutoff).astype(int)


X = df.iloc[:, 0:11] # Input variables
y = df['quality_binary'] # Output variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test.values) # Converting X_test to array using the values attribute  #.values: It returns the content of the DataFrame as a NumPy array.


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()  


# In[ ]:




