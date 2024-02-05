#!/usr/bin/env python
# coding: utf-8

# Project 2 Avocado Project

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load the dataset
avocado_data = pd.read_csv("https://github.com/dsrscientist/Data-Science-ML-Capstone-Projects/raw/master/avocado.csv")


# In[4]:


# Display the first few rows of the dataset
print(avocado_data.head())


# In[5]:


# Check the data types and missing values
print(avocado_data.info())


# In[6]:


# Summary statistics
print(avocado_data.describe())


# In[20]:


# Convert 'Date' column to datetime
avocado_data['Date'] = pd.to_datetime(avocado_data['Date'],dayfirst=True)


# In[8]:


# Visualizations
# Line plot for average price over time
plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='AveragePrice', data=avocado_data)
plt.title('Average Avocado Price Over Time')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.show()


# In[8]:


# Box plot for average price distribution by region
plt.figure(figsize=(12,6))
sns.boxplot(x='region', y='AveragePrice', data=avocado_data)
plt.title('Average Avocado Price Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.xticks(rotation=90)
plt.show()


# In[9]:


# Scatter plot for price vs. volume
plt.figure(figsize=(10,6))
sns.scatterplot(x='Total Volume', y='AveragePrice', data=avocado_data, alpha=0.5)
plt.title('Average Price vs. Total Volume')
plt.xlabel('Total Volume')
plt.ylabel('Average Price')
plt.show()


# Creates visualizations including a line plot showing the average avocado price over time, a box plot showing the average avocado price distribution by region, and a scatter plot showing the relationship between price and volume.

# In[10]:


# Visualizations
# Histogram of AveragePrice
plt.figure(figsize=(10, 6))
sns.histplot(avocado_data['AveragePrice'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Average Avocado Price')
plt.xlabel('Average Price ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[12]:


# Count plot of avocado types
plt.figure(figsize=(8, 6))
sns.countplot(data=avocado_data, x='type', palette='Set2')
plt.title('Distribution of Avocado Types')
plt.xlabel('Type')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# Creates a count plot showing the distribution of avocado types (conventional or organic).

# In[16]:


# Visualizations
# Line plot of Total Volume over the years
plt.figure(figsize=(10, 6))
sns.lineplot(data=avocado_data, x='year', y='Total Volume', estimator=sum, errorbar=None)
plt.title('Total Avocado Volume Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Volume')
plt.grid(True)
plt.show()


# In[17]:


# Box plot of Total Volume by region
plt.figure(figsize=(12, 8))
sns.boxplot(data=avocado_data, x='region', y='Total Volume')
plt.title('Total Avocado Volume by Region')
plt.xlabel('Region')
plt.ylabel('Total Volume')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# In[14]:


# Visualizations
# Line plot of Total number of avocados sold for different PLU codes over time
plt.figure(figsize=(10, 6))
plt.plot(avocado_data['Date'], avocado_data['4046'], label='PLU 4046')
plt.plot(avocado_data['Date'], avocado_data['4225'], label='PLU 4225')
plt.plot(avocado_data['Date'], avocado_data['4770'], label='PLU 4770')
plt.title('Total Avocado Sales for Different PLU Codes Over Time')
plt.xlabel('Date')
plt.ylabel('Total Number of Avocados Sold')
plt.legend()
plt.grid(True)
plt.show()


# These codes will load the dataset, display the first few rows, check data types and missing values, compute summary statistics, and create visualizations. The visualizations include a line plot showing the total number of avocados sold for different PLU codes over time and a scatter plot showing the relationship between total volume and average price, Total Avacado by region, Total Volume over the years and the organic or conventional type distibution. 
