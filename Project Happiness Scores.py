#!/usr/bin/env python
# coding: utf-8

# World Happiness project 
# 

# In[1]:


pip install scikit-learn


# In[2]:


#import necessary libararies 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


#import the dataset
happiness_data_url= "https://github.com/dsrscientist/DSData/raw/master/happiness_score_dataset.csv"
happiness_df=pd.read_csv(happiness_data_url)


# In[4]:


#print data
print(happiness_data_url)


# In[5]:


#print data
print(happiness_df.info())


# In[13]:


#Feature selection
Features=['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom']
x=happiness_df[Features]
y=happiness_df['Happiness Score']


# In[21]:


#split the dataset 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[30]:


#Model
model = LinearRegression()
model.fit(x_train, y_train)


# In[24]:


#predictions
y_pred=model.predict(x_test)


# In[28]:


#evluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[35]:


#importing libraries
import matplotlib.pyplot as plt


# In[36]:


#Visualize predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Happiness Score')
plt.ylabel('Predicted Happiness Score')
plt.title('Actual vs Predicted Happiness Score')
plt.show()


# In[38]:


#new data
plt.savefig('name of the file.png')


# In[39]:


# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({'Actual Happiness Score': y_test, 'Predicted Happiness Score': y_pred})


# In[40]:


# Display the DataFrame
print(results_df.head())


# In[52]:


#save the dataframe to csv
results_df.to_csv('actual_vs_predicted_happiness_scores.csv', index=False)


# In[ ]:




