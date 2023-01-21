import pandas as pd
import numpy as np
import copy
import random
import dataframe_image as dfi  

''' Importing the declared variables from previous steps'''

from Step1_DataPreprocessing import costs, tasks2, suppliers
from Step2_EDA import *
# Required variables from Step 1 & Step 2 are imported

''' 3. Machine Learning Model Fitting & Scoring'''
print("Section 3: Machine Learning Model Fitting & Scoring")

''' 3.1 Combine the task features, supplier features and costs into a single dataset '''

print("Section 3.1: Combine the task features, supplier features and costs into a single dataset")
# Merging datasets together
df_1 = pd.merge(costs, tasks2, how='inner', on='Task ID') 
df = pd.merge(df_1, suppliers,how='inner',on='Supplier ID')  
print('The number of rows in the costs dataframe were: ', str(len(costs)), \
      ' and the number of rows of the new dataframe is: ',str(len(df)))

# To order the columns in the same way as required 
cols_to_order = ['Task ID','Supplier ID'] #Columns to be placed at the beginning
cols_to_order2=['Cost'] #Columns to be placed at the end

# To avoid writing everything again and rearrange columns more efficiently
cols=cols_to_order+cols_to_order2
new_columns = cols_to_order + (df.columns.drop(cols).tolist()) + cols_to_order2
df = df[new_columns] #Create the ordered dataframe
df_copy=copy.deepcopy(df) #Before dropping the column of suppliers we create a copy
print("After changing columns horizontal indexing:")
print(df)

# Splitting the data
x=df.iloc[:,2:-1] #Task & Supplier features
y=df.iloc[:,-1] #Costs
Groups=df.iloc[:,0] #Task ID
print("Unique values in Groups: ", len(Groups.unique()))


''' 3.2 Splitting the Data in X and Y (train and Test) '''

print("Section 3.2: Splitting the Data in X and Y (train and Test)")
random.seed(50) #Set a seed to replicate results
sampletasks=Groups.unique() #Get a list with the unique Tasks (120)
TestGroup = random.sample(list(sampletasks), 20) #Randomly select 20
print("Length of TestGroup: ", len(TestGroup))

#Task ID included in TestGroup
tests=df[df['Task ID'].isin(TestGroup)].sort_values(by='Task ID').reset_index(drop=True) 

#Task ID NOT included in TestGroup
trains=df[~df['Task ID'].isin(TestGroup)].sort_values(by='Task ID').reset_index(drop=True) 

#x train (Tasks and Suppliers features of Tasks ID's that were NOT selected in the TestGroup)
x_train=trains.iloc[:,2:-1] 

#x test (Tasks and Suppliers features of Tasks ID's that were selected in the TestGroup)
x_test=tests.iloc[:,2:-1] 

#y train (Costs of Tasks ID's that were NOT selected in the TestGroup)
y_train=trains.iloc[:,-1] 

#y test (Costs of Tasks ID's that were selected in the TestGroup)
y_test=tests.iloc[:,-1] 

print("Unique values in tests['Task ID'] :",len(tests['Task ID'].unique()))
print("x_train values", x_train)


''' 3.3 Training and Testing Model (Lasso Regression & Ridge Regression) '''
print("Section 3.3: Training and Testing Model (Lasso Regression & Ridge Regression)")

''' 3.3.1 Lasso Model '''

print("Section 3.3.1: Lasso Model")

from sklearn.linear_model import Lasso # Importing Lasso Model from ScikitLearn

lasso = Lasso(alpha = 0.01, random_state=42, max_iter=10000)
lasso.fit(x_train, y_train)
pred_lasso = lasso.predict(x_test)
print(pred_lasso)
print("R2 score for Lasso Model: ", lasso.score(x_test,y_test)) #R2 score


''' 3.3.2 Ridge Model  '''

print("Section 3.3.2: Ridge Model")
# This is an additional model we have used to compare the result.
# As ridge and lasso are similar models, we are expecting to see the similar result. 
# But our intention is to see the differences between two models. 

from sklearn.linear_model import Ridge # Importing Ridge Model from ScikitLearn

ridge = Ridge(alpha = 1,random_state=42,max_iter=10000) 
ridge.fit(x_train, y_train)
pred_ridge = ridge.predict(x_test)
print(pred_ridge)
print("R2 score for Ridge Model: ", ridge.score(x_test,y_test)) #R2 score


''' 3.3.3 ElasticNet Model  '''
# Hybrid Model 

from sklearn.linear_model import ElasticNet

print("Section 3.3.3: ElasticNet Model")
Elnet=ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
Elnet.fit(x_train, y_train)
pred_ElasticNet = Elnet.predict(x_test)
print(pred_ElasticNet)
print("R2 score for ElasticNet Model: ", Elnet.score(x_test,y_test))
