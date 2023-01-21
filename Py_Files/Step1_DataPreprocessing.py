import pandas as pd
import numpy as np
import copy
import os

#Additional modules: exporting dataframes to png
import pip
pip.main(['install','dataframe_image', '--quiet'])
import dataframe_image as dfi  

# To make the Figures directory to store the graph images
if not os.path.exists('Figures'):
    os.mkdir('Figures')

import warnings
warnings.filterwarnings("ignore")

# Loading the given datasets
costs=pd.read_csv('./Support_Data/cost.csv')
suppliers=pd.read_csv('./Support_Data/suppliers.csv')
tasks=pd.read_excel('./Support_Data/tasks.xlsx')


''' Data Preprocessing '''

''' 1.1: Initial inspection and data correction '''

print("Section 1")
print("Section 1.1: Initial Inspection & Data Correction")

# To check the structure and the elements of the datasets: 
print("Structure & Elements of different datasets: ")
print(costs.head())
print(tasks.head())
print(suppliers.head())
print() # Line Gap

# DataFrame structure and data type 
print("Dataframe structure, non-null values and data types: ")
print(costs.info())     
print(tasks.info())     
print(suppliers.info()) 
print() # Line Gap

## Data Description
# Count the number of tasks, suppliers, features and cost values in all data sets

#tasks
print("'tasks' dataset contains:\n{} tasks\n{} task features".format(len(tasks),len(tasks.columns[1:])))
#suppliers
print("\n'suppliers' dataset contains:\n{} suppliers\n{} supplier features".format(len(suppliers),len(suppliers.columns[1:])))
#costs
print("\n'costs' dataset contains:\n{} tasks\n{} suppliers\n{} cost values".format(len(costs.loc[:,'Task ID'].unique()), \
    len(costs.loc[:,'Supplier ID'].unique()), len(costs.loc[:, 'Cost'])))
print() # Line Gap

# The format of the tasks ID do not match between tasks and costs dataframe.
# Both are objects (strings) : Change the Format to [dd-mm-yy]

# Before changing the format: 
print('Task ID format before changing datatype: ')
print(tasks['Task ID'].head(2))
print(costs['Task ID'].head(2))
print() # Line Gap

#To ensure deleting any whitespace from Task ID 
tasks['Task ID']=tasks['Task ID'].str.replace(' ', '')
costs['Task ID']=costs['Task ID'].str.replace(' ', '')
print() # Line Gap

#Format change
costs['Task ID']=pd.to_datetime(costs['Task ID'],infer_datetime_format=True, \
                                format='%d/%m/%y', dayfirst=True)

tasks['Task ID']=pd.to_datetime(tasks['Task ID'],infer_datetime_format=True, \
                                format='%d/%m/%y', dayfirst=True)

# After changing the format:
print('Task ID format after changing datatype: ')
print(tasks['Task ID'].head(2))
print(costs['Task ID'].head(2))
print() # Line Gap

# To check the missing Values:
print('Number of NA values in suppliers dataset: ', suppliers.isna().sum().sum())
print('Number of NA values in costs dataset: ', costs.isna().sum().sum())
print('Number of NA values in tasks dataset: ', tasks.isna().sum().sum())
print() # Line Gap

# Eliminate tasks that does not have a cost related to
print('Number of tasks in the tasks dataset:',tasks['Task ID'].nunique(), 
      '\nNumber of tasks in the Costs dataset:',costs['Task ID'].nunique())
print() # Line Gap

#We get an array with all the tasks present in the costs table
costs_id=costs['Task ID'].unique() 
tasks=tasks.loc[tasks['Task ID'].isin(costs_id)] 

print('Number of tasks in task dataset(Excluding Task IDs that are not in Costs):', len(tasks)) 
print() # Line Gap

''' 1.2: Features with very low variance '''

print("Section 1.2: Features with very low variance ")
# As per the instruction, calculate the maximum value, minimum value, mean and variance of each feature. 
# List of Variance for Tasks 
lisT = []
for column in tasks.iloc[:,1:].columns:
    lisT.append(np.var(tasks[column]))               

# List of Variance for Suppliers
lisS = []
for column in suppliers.iloc[:,1:].columns:
    lisS.append(np.var(suppliers[column]))   

#To present result in floating format 0.03f 
pd.options.display.float_format = '{:.3f}'.format

descriptives=round(tasks.describe().loc[['max','min','mean','std'],:], 4)
descriptives.loc['variance'] = lisT
print(descriptives.T)

descriptivesS=round(suppliers.describe().loc[['max','min','mean','std'],:], 4)
descriptivesS.loc['variance'] = lisS
print(descriptivesS.T)

#To reset the floating format as normal
pd.reset_option('display.float_format')

# 1 Tasks

## Drop the features with lower variance
to_drop = [column for column in tasks.iloc[:,1:].columns \
           if (np.var(tasks[column]) <= 0.0100)]
tasks.drop(to_drop, axis=1, inplace=True)
print("Number of columns to drop from tasks dataset: ", len(to_drop))
# Total number of columns changed from 106 to 82 
print("Total number of columns changed from 106 to :", len(tasks.columns))

# 2 Suppliers
to_drop = [column for column in suppliers.iloc[:,1:].columns \
           if (np.var(suppliers[column]) <= 0.0100)]
tasks.drop(to_drop, axis=1, inplace=True)
print("Number of columns to drop from suppliers dataset: ", len(to_drop))
# Total number of supplier columns did not change
print("Total number of columns remains same :", len(suppliers.columns))
print() # Line Gap

''' 1.3 Scaling from -1 to 1 '''

print("Section 1.3: Scaling from -1 to 1 ")

# MinMaxScaler was used to scale the data values to a range of -1 to 1.
#Create the object "scaler"
from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(-1, 1)) 

#To avoid changing the first column (ID), "iloc" is used
tasks.iloc[:,1:] = scaler.fit_transform(tasks.iloc[:,1:])
print(tasks.head())

# Suppliers
# As per the insturction "conversion to all feature", supplier features were also scaled. 
suppliers.iloc[:,1:]=scaler.fit_transform(suppliers.iloc[:,1:]) 
print(suppliers.head())
print() # Line Gap

''' 1.4 Correlation (just for tasks) '''
print("Section 1.4: Correlation (tasks)")

# First, we have populated the heatmap which was created manually using dataframe style with the entire datasets of task.

tasks_corr = tasks.iloc[:,1:].corr().abs() 
tasks_corr.style.background_gradient(cmap='coolwarm',axis=None).format(precision=2) 
#Converting correlation matrix in a heatmap. 

masks = np.zeros_like(tasks_corr, dtype=bool)
masks[np.triu_indices_from(masks)] = True 
#To avoid redundancy of the matrix (no repeated correlation coefficients)
tasks_corr[masks] = np.nan 
# Because there is error on null_color hyper parameter due to different pandas version, 
# we used try-except to make it work in any version.
try:
         tasks_corr_styled=(tasks_corr
                       .style
                       .format(precision=2)
                       .background_gradient(cmap='coolwarm', axis=None, vmin=0, vmax=1)
                       .highlight_null(color='#f1f1f1'))
except: 
         tasks_corr_styled=(tasks_corr
                       .style
                       .format(precision=2)
                       .background_gradient(cmap='coolwarm', axis=None, vmin=0, vmax=1)
                       .highlight_null(null_color='#f1f1f1'))
print(tasks_corr_styled)

#Exporting PNG file of the task_corr_styled
#dfi: DataFrame_Image Object 
dfi.export(tasks_corr_styled, "Figures/1.4_corr1.png", max_cols=-1, max_rows=-1)
print("Section 1.5: Correlation Matrix is saved in Figures directory")

# The following lines are customization of heatmap, created manually using dataframe. 
# A deepcopy to apply changes safely :)
tasks2=copy.deepcopy(tasks) 

for i in range(len(tasks2.columns)):
    corr=pd.DataFrame(tasks2.iloc[:,1:].corr().abs().unstack()).reset_index()\
    .rename(columns={'level_0': 'FeatureGroup1', 'level_1': 'FeatureGroup2',0:'Corr'})  
    #.corr().abs(): Calculation of absolute correlation. 
    #.unstack: Turn all index to Column ()
    #.reset_index: Reset the index because we will need the variable 1 for groupping. 
    #.rename: Column name renaming to Feature_Group1, Feature_Group2 and Corr

    corr['Number']=corr.Corr>=0.8 
    #New column: Corr is larger than or equal to 0.8(True) or not(False).
    
    corr_group=corr.groupby('FeatureGroup1').sum('Number').sort_values('Number',ascending=False).head(1) 
    #.Groupby 'Feature_Group1' 
    #.sum('Number'): sum of True(greater than 0.8)
    #.sort_values.head(1): sort the value and get the first row only.
    
    lista=corr_group.index.tolist() 
    #Store the row(feature) with greater than or equal 0.8 corr. 
    
    if corr_group.Number[0]>1: 
        tasks2=tasks2.drop(columns=lista) 
        #Drop the column using stored the feature(lista)

print("After the process, the number of features in tasks dataset changed from ",\
      len(tasks.iloc[:,1:].columns),"to ",len(corr['FeatureGroup1'].unique()))

tasks_corr = tasks2.iloc[:,1:].corr().abs() 
tasks_corr.style.background_gradient(cmap='coolwarm',axis=None).format(precision=2)
masks = np.zeros_like(tasks_corr, dtype=bool)
masks[np.triu_indices_from(masks)] = True
tasks_corr[masks] = np.nan

# Because there is an error on null_color hyper parameter due to different pandas versions, 
# we used try-except to make it work in any pandas version. 
try: 
    tasks_corr_styled2=(tasks_corr
                        .style
                        .format(precision=2)
                        .background_gradient(cmap='coolwarm', axis=None, vmin=0, vmax=1) 
                         #We adjust pallete to go from 0 to 1
                        .highlight_null(color='#f1f1f1')) # NaNs Color grey
except: 
    tasks_corr_styled2=(tasks_corr
                    .style
                    .format(precision=2)
                    .background_gradient(cmap='coolwarm', axis=None, vmin=0, vmax=1) 
                     #We adjust pallete to go from 0 to 1
                    .highlight_null(null_color='#f1f1f1')) # NaNs Color grey

print(tasks_corr_styled2)

#Export png file of the styled dataframe
dfi.export(tasks_corr_styled2, "Figures/1.4_corr2.png", max_cols=-1, max_rows=-1)
print("Section 1.5: Customized Correlation Matrix is saved in Figures directory")
print() # Line Gap


''' 1.5 Identify the top 20 suppliers for each task '''

print("Section 1.5: Identify the top 20 suppliers for each task")

# Establish the index for better readability
costs1=costs.set_index(['Task ID','Supplier ID']).sort_values(by=['Task ID','Cost']) 
# Get ranking of cost(Supplier) by Task ID (ascending order)
costs1['Ranking']=costs1.groupby('Task ID')['Cost'].rank() 
# To check the ranking is applied to all tasks
print(costs1.head(68)) 

# cost dataframe with only top 20 suppliers by each Task ID
costs_top=costs1[costs1['Ranking']<21].reset_index()
# Array of any suppliers included in the top 20 ranking
sup_top=costs_top['Supplier ID'].unique() 

print("Appeared one or more times in TOP 20 ranking:",len(sup_top), \
      "out of total suppliers",len(costs['Supplier ID'].unique()))

# We only keep the top suppliers in the suppliers dataset. In other words, we drop the 'always expensive' supplier.

# Cost DataFrame
print("Original supplier number in Cost DataFrame:", len(costs))
# Keep 63 supplier appeared in top 20 ranking
costs=costs[costs['Supplier ID'].isin(sup_top)]
print("After removing always expensive supplier(Top 20):", len(costs))

# Supplier DataFrame
print('Original supplier number in Suppliers DataFrame:', len(suppliers))
# Keep 63 supplier appeared in top 20 ranking
suppliers=suppliers[suppliers['Supplier ID'].isin(sup_top)] 
print('After removing always expensive supplier(Top 20):', len(suppliers))
print() # Line Gap

# Reseting the index of new datasets to avoid confusions
costs.reset_index(drop=True, inplace=True) 
tasks2.reset_index(drop=True, inplace=True)
suppliers.reset_index(drop=True, inplace=True)
