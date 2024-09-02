#!/usr/bin/env python
# coding: utf-8

# # Feature Selection Using Scikit-Learn

# In this exercise, you will perform feature selection using built-in scikit-learn tools.
# * You will train decision tree models on "cell2cell," a telecom company churn prediction data set.
# * After performing a grid search to find the hyperparameter configuration that results in the best performing Decision Tree, you will perform feature selection to find the most important features in your training data for predicting customer churn.
# 
# **<font color='red'>Note: Some of the code cells in this notebook may take a while to run.</font>**

# ### Import Packages
# 
# Before you get started, import a few packages. Run the code cell below. 

# In[1]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# We will also import the scikit-learn `DecisionTreeClassifier`, the `train_test_split()` function for splitting the data into training and test sets, `GridSearchCV` to perform model selection to find the model with the best cross-validation score, and the metric `accuracy_score` to evaluate your model.

# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# ## Step 1: Build Your DataFrame and Define Your ML Problem
# 

# We will work with the "cell2celltrain" data set. This data set is already preprocessed, with the proper formatting, outliers and missing values taken care of, and all numerical columns scaled to the [0, 1] interval. One-hot encoding has been performed on all categorical columns. It is ready for modeling. Run the cell below to load the data set and save it to DataFrame `df`.

# In[3]:


filename = os.path.join(os.getcwd(), "data", "cell2celltrain.csv")
df = pd.read_csv(filename, header=0)


# #### Define the Label
# 
# This is a binary classification problem in which we will predict customer churn. The label is the `Churn` column.
# 
# #### Identify Features
# 
# For now, our features will be all of the remaining columns in the dataset. After performing feature selection, we may choose to train our model on fewer features.

# ## Step 2. Create Labeled Examples from the Data Set 

# In[4]:


y = df['Churn']
X = df.drop(columns = 'Churn', axis=1)
X.head()

print(type(X))


# ## Step 3: Create Training and Test Data Sets
# The code cell below creates training and test data sets. Since we will be performing model selection, we will split our 10% of our data to serve as a test set.

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)


# ## Step 4: Perform Decision Tree Model Selection Using Grid Search
# 

# We will use scikit-learn's `GridSearchCV` to search over different combinations of possible hyperparameter values to find the set that results in the best cross-validation (CV) score. We will look for the best values of the following hyperparameters: the maximum depth of the tree (`max_depth`) and the minimum number of samples required to be at a leaf node (`min_samples_leaf`). 

# ### a. Set up the Parameter Grid

# In[6]:


md = [2**n for n in range(2,6)]
msl = [25*2**n for n in range(0,3)]
param_grid={'max_depth':md, 'min_samples_leaf':msl}

param_grid


# ### b. Perform Grid Search Cross-Validation
# <b>Note</b>: The cell below may take a little while to run.
# 

# In[7]:


print('Running Grid Search...')

# 1. Create a DecisionTreeClassifier model object without supplying arguments
model = DecisionTreeClassifier()

# 2. Run a Grid Search with 5-fold cross-validation using our the model.
#   Pass all relevant parameters to GridSearchCV and assign the output to the object 'grid'
grid = GridSearchCV(model, param_grid, cv=5)

# 3. Fit the model to the training data and assign the fitted model to the 
#    variable grid_search
grid_search = grid.fit(X_train, y_train)

print('Done')


# ### c. Identify the Best Hyperparameter Values
# 
# In the code cell below, access the best hyperparameter values for `max_depth` and `min_samples_leaf` from the `grid_search` object. *<b>Hint</b>*: You can access specific hyperparameter values using the `grid_search` object's `best_estimator_` attribute followed by the hyperparameter name (use dot notation to connect the object, attribute, and hyperparameter named as `object.attribute.name`).
# 
# 1. Save the best value of `max_depth` to variable `best_md`.
# 2. Save the best value of `min_samples_leaf` to variable `best_msl`.

# ### Graded Cell
# 
# The cell below will be graded. Remove the line "raise NotImplementedError()" before writing your code.

# In[8]:


# YOUR CODE HERE
best_md = grid_search.best_estimator_.max_depth
best_msl = grid_search.best_estimator_.min_samples_leaf

# Print best performing hyperparameter configuration
print('Best value for max depth: {0}'.format(best_md))
print('Best value for the minimum number of samples required to be at a leaf node: {0}'
      .format(best_msl))

# Print best accuracy score resulting from this configuration
print('Accuracy score: {0}'.format(grid_search.best_score_))


# ### Self-Check
# 
# Run the cell below to test the correctness of your code above before submitting for grading. Do not add code or delete code in the cell.

# In[9]:


# Run this self-test cell to check your code; 
# do not add code or delete code in this cell
from jn import testGridSearchValues

try:
    p, err = testGridSearchValues(best_md, best_msl)
    print(err)
except Exception as e:
    print("Error!\n" + str(e))
    


# ## Step 5. Train the Best Model

# Now that you have found the values of `max_depth` and `min_samples_leaf` that produce the best model, you can fit a `DecisionTreeClassifier` with those hyperparameter values. 
# 
# In the code cell below, fit the *one* decision tree classifier using the best hyperparameters identified, test your model on the test set (`X_test`), and obtain the final accuracy score of your model's class label predictions.

# ### Graded Cell
# 
# The cell below will be graded. Remove the line "raise NotImplementedError()" before writing your code.

# In[10]:


# 1. Create a DecisionTreeClassifier model object and assign it to the variable 'model'
model = DecisionTreeClassifier(max_depth=best_md, min_samples_leaf=best_msl)
    
# 2. Fit the model to the training data 
model.fit(X_train, y_train)

# 3. Use the predict() method to make predictions on the test data and assign the results to the variable
# 'class_label_predictions'
class_label_predictions = model.predict(X_test)

# 4. Compute the accuracy score and assign the result to the variable 'acc_score'
acc_score = accuracy_score(y_test, class_label_predictions)

print('Accuracy score: {0}'.format(acc_score))


# ### Self-Check
# 
# Run the cell below to test the correctness of your code above before submitting for grading. Do not add code or delete code in the cell.

# In[11]:


# Run this self-test cell to check your code; 
# do not add code or delete code in this cell
from jn import testDTModel

try:
    p, err = testDTModel(df, model,class_label_predictions, acc_score)
    print(err)
except Exception as e:
    print("Error!\n" + str(e))
    


# ## Step 6: Perform  Feature Selection Using Scikit-learn

# In the code cell below, you will find the five most important features in your training data (`X_train`) for predicting churn.
# 
# Perform the following steps:
# 
# 1. Through your `model` object, you can obtain scores corresponding to the importance of the predictive features on which the examples were split for classification using the `feature_importances_` attribute. You can use this attribute with the following syntax: `model.feature_importances_`. This returns a NumPy array. Use this method to obtain an array of scores and save the array to the variable `feature_imp`.
# 
# 2. Create a Pandas DataFrame out of all feature names and their measures of importance by using the `pd.DataFrame()` function. Call the function with a dictionary containing the following key/value pairs:
#     * `'name': X_train.columns.values`
#     * `'imp': feature_imp`
#     
#     Assign the DataFrame to the variable `df_features`.
# 
# 3. Using the Pandas method `sort_values()`, sort the importance scores in the `imp` column in the new  DataFrame `df_features` in descending order. Assign the resulting DataFrame to variable `df_sorted`.
# 
# 4. Using `iloc`, extract the top 5 values in `df_sorted`. Then, extract the values in the column `name` in `df_sorted`. Save these names to a Python list called `top_five `.
# 

# ### Graded Cell
# 
# The cell below will be graded. Remove the line "raise NotImplementedError()" before writing your code.

# In[13]:


#1. Obtain "feature importance" scores from the model object and save the array to the variable 
# 'feature_imp'

feature_imp = model.feature_importances_

# 2. Create a Pandas DataFrame with a list of all features and their scores. 
# Save the result to the variable 'df_features'
df_features = pd.DataFrame({'name': X_train.columns.values, 'imp': feature_imp})

# 3. Sort df_features in descending order and
# save the result to the variable 'df_sorted'
df_sorted = df_features.sort_values(by='imp', ascending=False)

# 4. Obtain the top 5 sorted feature names and save the result to list 'top_five' 
top_five = df_sorted.iloc[:5]['name'].tolist()

print('Top five features: {0}'.format(top_five))


# ### Self-Check
# 
# Run the cell below to test the correctness of your code above before submitting for grading. Do not add code or delete code in the cell.

# In[14]:


# Run this self-test cell to check your code; 
# do not add code or delete code in this cell
from jn import testHFS

try:
    p, err = testHFS(df, model,feature_imp, df_features, df_sorted, top_five)
    print(err)
except Exception as e:
    print("Error!\n" + str(e))
    


# The code cell below produces a bar plot for the normalized importance of the top 5 most predictive features on which the examples were split for classification.

# In[15]:


fig, ax = plt.subplots()
ax.bar(np.arange(5), sorted(model.feature_importances_, reverse=True)[:5], width = 0.35)
ax.set_xticks(np.arange(5))
ax.set_xticklabels(top_five, rotation = 90)
plt.title('Feature importance from DT')
ax.set_ylabel('Normalized importance')


# We can train our model using only the most important features and see whether the resulting accuracy score will increase. We can also experiment with different combinations of features, such as removing the least important feature.
