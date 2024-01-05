#!/usr/bin/env python
# coding: utf-8

# Titianic Survived Project

# In[13]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[14]:


# Load the Titanic dataset
titanic_data_url = "https://github.com/dsrscientist/dataset1/raw/master/titanic_train.csv"
titanic_df = pd.read_csv(titanic_data_url)


# In[16]:


#print data
print(titanic_data_url)


# In[18]:


# Handle missing values (for simplicity, you can customize this based on your data exploration)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)


# In[19]:


# Feature selection (considering relevant features)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = pd.get_dummies(titanic_df[features], columns=['Sex', 'Embarked'], drop_first=True)
y = titanic_df['Survived']


# In[20]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[22]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[23]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


# In[ ]:




