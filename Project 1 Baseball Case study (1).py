#!/usr/bin/env python
# coding: utf-8

# Project 1 Base Ball Case Study

# In[1]:


pip install pandas scikit-learn


# In[12]:


#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[13]:


#load the data set
url= "https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv"
data=pd.read_csv(url)


# In[14]:


#print the first few rows of dataset
print(data.head())


# In[15]:


# Check for missing values
print(data.isnull().sum())


# In[11]:


# Split the data into features (X) and target variable (y)
X = data.drop('W', axis=1)  # Features
y = data['W']               # Target variable


# In[16]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[18]:


# Predict on the test set
y_pred = model.predict(X_test)


# In[19]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[20]:


print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[21]:


import matplotlib.pyplot as plt


# In[23]:


# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual vs Predicted Wins')
plt.show()


# In[24]:


# Plotting the residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Wins')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()


# In this plot, the x-axis represents the actual number of wins, and the y-axis represents the residuals. A horizontal line at y=0 indicates where the residuals should ideally lie. Points scattered around this line indicate how the model's predictions deviate from the actual values.

# These visualizations provide insights into the model's performance and can help identify any patterns or trends in the predictions and residuals.

# In[25]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[26]:


# Prepare the Data
X = data[['R']]  # Selecting 'Runs scored' as the feature
y = data['W']    # Number of wins as the target variable


# In[27]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[29]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[30]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Runs scored')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


#  3.	AB - This means At bat or time at bat. It's is a batter's turn batting against a pitcher: plate appearances, not including bases on balls, being hit by pitch, sacrifices, interference, or obstruction.

# In[31]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[32]:


# Prepare the Data
X = data[['AB']]  # Selecting 'At bat' as the feature
y = data['W']     # Number of wins as the target variable


# In[33]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[35]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[36]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('At bat')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


#  4.	H - This means Hit. It's also called a "base hit", is credited to a batter when the batter safely reaches or passes first base after hitting the ball into fair territory, without the benefit of either an error or a fielder's choice: reaching base because of a batted, fair ball without error by the defense.

# In[37]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[38]:


# Prepare the Data
X = data[['H']]  # Selecting 'Hit' as the feature
y = data['W']    # Number of wins as the target variable


# In[39]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[41]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[42]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Hits')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# 5.	2B - This means the act of a batter striking the pitched ball and safely reaching second base without being called out by the umpire, without the benefit of a fielder's misplay (see error) or another runner being put out on a fielder's choice. A double is a type of hit (the others being the single, triple and home run) and is sometimes called a "two-bagger" or "two-base hit": hits on which the batter reaches second base safely without the contribution of a fielding error.

# In[43]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[44]:


# Prepare the Data
X = data[['2B']]  # Selecting 'Doubles' as the feature
y = data['W']     # Number of wins as the target variable


# In[45]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[47]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[48]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Doubles')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# 6.	3B - This measns a Triple.It's is the act of a batter safely reaching third base after hitting the ball, with neither the benefit of a fielder's misplay nor another runner being put out on a fielder's choice. A triple is sometimes called a "three-bagger" or "three-base hit": hits on which the batter reaches third base safely without the contribution of a fielding error.

# In[49]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[50]:


# Prepare the Data
X = data[['3B']]  # Selecting 'Triples' as the feature
y = data['W']     # Number of wins as the target variable


# In[51]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[52]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[53]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[54]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Triples')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This code incorporates the '3B' (triples) feature into our analysis. We load the dataset, select 'Triples' as the input feature, 'Number of wins' (W) as the target variable, train a linear regression model, evaluate its performance, and visualize the results.

# 7.	HR - This means Home runs. It's scored when the ball is hit in such a way that the batter is able to circle the bases and reach home plate safely in one play without any errors being committed by the defensive team. A home run is usually achieved by hitting the ball over the outfield fence between the foul poles (or hitting either foul pole) without the ball touching the field: hits on which the batter successfully touched all four bases, without the contribution of a fielding error.

# In[55]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[56]:


# Prepare the Data
X = data[['HR']]  # Selecting 'Home Runs' as the feature
y = data['W']     # Number of wins as the target variab


# In[57]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[59]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[60]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Home Runs')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# 8.	BB - This means Base on balls (also called a "walk"). It occurs in baseball when a batter receives four pitches that the umpire calls balls, and is in turn awarded first base without the possibility of being called out: hitter not swinging at four pitches called out of the strike zone and awarded first base.

# In[61]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[62]:


# Prepare the Data
X = data[['BB']]  # Selecting 'Base on balls' as the feature
y = data['W']     # Number of wins as the target variable


# In[63]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[64]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[65]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[66]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Base on Balls')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# 9.	SO - Also denoted as "K" means Strikeout. It occurs when a batter accumulates three strikes during a time at bat. It usually means that the batter is out: number of batters who received strike three.

# In[69]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[68]:


# Prepare the Data
X = data[['SO']]  # Selecting 'Strikeouts' as the feature
y = data['W']     # Number of wins as the target variable


# In[70]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[71]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[72]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[73]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Strikeouts')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This information incorporates the 'SO' (strikeout) feature into our analysis. We load the dataset, select 'Strikeouts' as the input feature, 'Number of wins' (W) as the target variable, train a linear regression model, evaluate its performance, and visualize the results.

# 10.	SB - This means Stolen base. It occurs when a runner advances to a base to which they are not entitled and the official scorer rules that the advance should be credited to the action of the runner: number of bases advanced by the runner while the ball is in the possession of the defense.

# In[74]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[75]:


# Prepare the Data
X = data[['SB']]  # Selecting 'Stolen Bases' as the feature
y = data['W']     # Number of wins as the target variable


# In[76]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[77]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[78]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[79]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Stolen Bases')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This code snippet incorporates the 'SB' (stolen bases) feature into our analysis. We load the dataset, select 'Stolen Bases' as the input feature, 'Number of wins' (W) as the target variable, train a linear regression model, evaluate its performance, and visualize the results.

# 11.	RA - This means Run Average. It refer to measures of the rate at which runs are allowed or scored.

# In[80]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[81]:


# Prepare the Data
X = data[['RA']]  # Selecting 'Run Average' as the feature
y = data['W']     # Number of wins as the target variable


# In[82]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[83]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[84]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[85]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Run Average')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This code snippet includes the 'RA' (Run Average) feature into our analysis. We load the dataset, select 'Run Average' as the input feature, 'Number of wins' (W) as the target variable, train a linear regression model, evaluate its performance, and visualize the results.

# 12.	ER - This means Earned run. It refers to any run that was fully enabled by the offensive team's production in the face of competent play from the defensive team: number of runs that did not occur as a result of errors or passed balls

# In[86]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[87]:


# Prepare the Data
X = data[['RA']]  # Selecting 'Run Average' as the feature
y = data['W']     # Number of wins as the target variable


# In[88]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[89]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[90]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[91]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Run Average')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This code snippet includes the 'RA' (Run Average) feature into our analysis. We load the dataset, select 'Run Average' as the input feature, 'Number of wins' (W) as the target variable, train a linear regression model, evaluate its performance, and visualize the results.

# 12.	ER - This means Earned run. It refers to any run that was fully enabled by the offensive team's production in the face of competent play from the defensive team: number of runs that did not occur as a result of errors or passed balls.

# In[92]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[93]:


# Prepare the Data
X = data[['ER']]  # Selecting 'Earned Run' as the feature
y = data['W']     # Number of wins as the target variable


# In[95]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[96]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[97]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[98]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Earned Run')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This code snippet includes the 'ER' (Earned Run) feature into our analysis. We load the dataset, select 'Earned Run' as the input feature, 'Number of wins' (W) as the target variable, train a linear regression model, evaluate its performance, and visualize the results.

# 13.	ERA - This means Earned Run Average. It refers to the average of earned runs allowed by a pitcher per nine innings pitched (i.e. the traditional length of a game). It is determined by dividing the number of earned runs allowed by the number of innings pitched and multiplying by nine: total number of earned runs (see "ER" above), multiplied by 9, divided by innings pitched.

# In[99]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[103]:


# Display the first few rows of the dataset to verify the addition of 'IPouts'
print(data.head())


# In[ ]:


# Calculate IPouts for column named 'IP'
data['IPouts'] = data['IP'].apply(lambda x: int(x.split('.')[0]) * 3 + int(x.split('.')[1]) if '.' in x else int(x) * 3)


# In[ ]:


# Add IPouts to the existing IP column
data['IP'] = data['IP'] + data['IPouts']


# In[110]:


# Verify the changes
print(data.head())


# In[ ]:


# Prepare the Data
data['ERA'] = (data['ER'] * 9) / data['IPouts']  # Calculate ERA
X = data[['ERA']]  # Selecting 'ERA' as the feature
y = data['W']      # Number of wins as the target variable


# In[112]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[113]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[114]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[115]:


# Visualize the Results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('ERA')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# We calculate the 'ERA' feature using the provided formula.
# Then, we proceed with the usual steps of preparing the data, splitting it into training and testing sets, choosing and training a linear regression model, evaluating its performance, and visualizing the results. Adjust the code as necessary for your analysis.

# 14.	CG - This means Complete Game. It's the act of a pitcher pitching an entire game without the benefit of a relief pitcher. A pitcher who meets this criterion will be credited with a complete game regardless of the number of innings played: number of games where player was the only pitcher for their team.

# In[116]:


# Explore the Data
print(data.head())
print(data.info())
print(data.describe())


# In[117]:


# Prepare the Data
X = data[['CG']]  # Selecting 'Complete Game' as the feature
y = data['W']     # Number of wins as the target variable


# In[118]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[119]:


# Choose a Model and Train it
model = LinearRegression()
model.fit(X_train, y_train)


# In[120]:


# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[121]:


# Scatter plot of actual vs. predicted wins
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual vs. Predicted Wins (Linear Regression)')
plt.show()


# creates a scatter plot where each point represents a team's actual wins (on the x-axis) against the predicted wins by the model (on the y-axis). The red dashed line represents a perfect prediction (where actual wins equal predicted wins). If the points align closely to this line, it indicates that the model's predictions are accurate.

# 15.	SHO - This means Shutout. It refers to the act by which a single pitcher pitches a complete game and does not allow the opposing team to score a run: number of complete games pitched with no runs allowed.

# In[123]:


# Display the first few rows of the dataset
print(data.head())


# In[124]:


# Check for missing values
print(data.isnull().sum())


# In[125]:


# Prepare the data
X = data[['SHO']]  # Selecting 'Shutout' as the feature
y = data['W']      # Number of wins as the target variable


# In[126]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[127]:


# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[128]:


# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[129]:


# Visualize the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Shutout')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This code will load the dataset, prepare the data by selecting the "Shutout" feature as the input (X) and the number of wins as the target variable (y), split the data into training and testing sets, train a linear regression model, evaluate the model's performance using mean squared error, mean absolute error, and R-squared score, and finally visualize the relationship between "Shutout" and the number of wins using a scatter plot.

# 16.	SV - This means Save. It's credited to a pitcher who finishes a game for the winning team under certain prescribed circumstances: number of games where the pitcher enters a game led by the pitcher's team, finishes the game without surrendering the lead, is not the winning pitcher, and either (a) the lead was three runs or fewer when the pitcher entered the game; (b) the potential tying run was on base, at bat, or on deck; or (c) the pitcher pitched three or more innings.

# In[130]:


# Display the first few rows of the dataset
print(data.head())


# In[131]:


# Check for missing values
print(data.isnull().sum())


# In[132]:


# Prepare the data
X = data[['SV']]  # Selecting 'Save' as the feature
y = data['W']     # Number of wins as the target variable


# In[133]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[134]:


# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[135]:


# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[136]:


# Visualize the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Save')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# This code will load the dataset, prepare the data by selecting the "Save" feature as the input (X) and the number of wins as the target variable (y), split the data into training and testing sets, train a linear regression model, evaluate the model's performance using mean squared error, mean absolute error, and R-squared score, and finally visualize the relationship between "Save" and the number of wins using a scatter plot.

# 17.	E - This means Errors. It's an act, in the judgment of the official scorer, of a fielder misplaying a ball in a manner that allows a batter or baserunner to advance one or more bases or allows a plate appearance to continue after the batter should have been put out. The term error is sometimes used to refer to the play during which an error was committed: number of times a fielder fails to make a play he should have made with common effort, and the offense benefits as a result.

# In[137]:


# Display the first few rows of the dataset
print(data.head())


# In[138]:


# Check for missing values
print(data.isnull().sum())


# In[139]:


# Prepare the data
X = data[['E']]  # Selecting 'Errors' as the feature
y = data['W']    # Number of wins as the target variable


# In[140]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[141]:


# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[142]:


# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)


# In[143]:


# Visualize the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Errors')
plt.ylabel('Number of Wins')
plt.title('Linear Regression Model')
plt.show()


# Prepare the data by selecting the "Errors" feature as the input (X) and the number of wins as the target variable (y), split the data into training and testing sets, train a linear regression model, evaluate the model's performance using mean squared error, mean absolute error, and R-squared score, and finally visualize the relationship between "Errors" and the number of wins using a scatter plot.
