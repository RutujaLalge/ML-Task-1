#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Sparks Foundation

# ## Data Science & Business Analytics Internship

# ### Task - 1 Prediction using Supervised ML

# #### In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.

# ## Author : Rutuja Vinod Lalge

# ### Batch - GRIPJAN22

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 1 - Importing  and Reading Data

# In[2]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# ### Step 2 - Plotting the Data

# In[3]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### Step 3 -  Preparing the data

# In[4]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# ### Step 4 - We will divide the data for training and testing the model

# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### Step 5 - Training the Algorithm

# In[6]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# ### Step 6 - Plotting the Line of Regression

# In[7]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### Step 7 - Making the Predictions

# In[8]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[9]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[13]:


# plotting the Bar graph to find the difference between the Actual and Predicted value
df.plot(kind='bar',figsize=(5,5))
plt.grid()


# ### Step 8 - To find out what will be the predicted score if a student studies for 9.25 hrs/day?

# In[15]:


# Testing with our own data
hours = np.array([[9.25]])
prediction = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(prediction[0]))


# ### Step 9 - Evaluating the Model

# In[11]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# #### According to the regression model, if a student studies for 9.25 hrs/day then the predicted score will be 93.69

#                                                       THANK YOU
