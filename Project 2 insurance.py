#!/usr/bin/env python
# coding: utf-8

# Project 2 Medical insurance cost 

# In[26]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[27]:


# extracting the dataset locally (replace "path_to_downloaded_file" with the actual path)
local_path = "path_to_downloaded_file/medical_cost_insurance.csv"
df = pd.read_csv(local_path)


# In[29]:


# My code that I have written from my end for the medical insurance project 


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset locally (replace "path_to_downloaded_file" with the actual path)
local_path = "path_to_downloaded_file/medical_cost_insurance.csv"
df = pd.read_csv(local_path)

# Explore the dataset
print(df.head())

# Encode variables (if needed)
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Define input features and output variable
X = df_encoded.drop('charges', axis=1)  # Input features
y = df_encoded['charges']  # Output variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Medical Costs')
plt.show()


# In[ ]:




