import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import matplotlib as pltlib

## Import modules for EDA
from sklearn.metrics import mean_squared_error
from math import sqrt

#Additional modules to install natsort:
import pip
pip.main(['install','natsort', '--quiet'])
import natsort as ns

''' Importing the declared variables from Step 1 Data Preprocessing '''
from Step1_DataPreprocessing import tasks2, costs 
# Required variables are only imported for the next step to avoid memory leak
# And, to avoid repetition of import statements

''' 2. Exploratory Data Analysis '''

print("Section 2: Exploratory Data Analysis (EDA)")

''' 2.1.1 Distribution of tasks by task features '''

print("Section 2.1.1: Boxplot showing Distribution of tasks by task features")

# This graph allows us to notice the difference among tasks for each task feature.
plt.rcParams.update(plt.rcParamsDefault) #Line to reset everything in Matplotlib to default
plt.style.use('tableau-colorblind10') 
plt.figure(figsize=(20,10)) 
fig1=sns.boxplot(data=tasks2) 
fig1.set(xlabel='Features') 
fig1.axes.set_title('Distribution of Tasks according to features',fontsize=25)
plt.tight_layout() 
plt.savefig("./Figures/2.1.1_Feature_Dist_Box.png")
print("Section 2.1.1: Boxplot Graph is saved in Figures directory")
plt.show() 


''' 2.1.2 Boxplot showing distribution of feature values for each task 
- Distribution of task feature values by task   '''


print("Section 2.1.2: Boxplot showing distribution of feature values for each task")

#Plotting distributions of tasks2 dataset
col_names = []
for col in tasks2.columns:
    col_names.append(col)
tasks3  = tasks2.melt(id_vars= 'Task ID', value_vars= col_names[1:],\
                      value_name= 'Task Feature Values', var_name = 'Task Feature')
plt.rcParams.update(plt.rcParamsDefault) #Line to reset everything in Matplotlib to default
plt.style.use('tableau-colorblind10') #Style of the plot
plt.figure(figsize=(200,50)) #Figsize of the plot

fig1=sns.boxplot(data=tasks3, x = "Task ID", y = "Task Feature Values",width=0.85,
                 medianprops=dict(color="blue", alpha=0.7),
                 flierprops=dict(markerfacecolor="#707070", marker="d")) #Plotting
for _,s in fig1.spines.items():
    s.set_linewidth(10)
    s.set_color('cyan')
sns.despine(offset=10, trim=True)
ax = fig1.axes  # Plotting axes
# From the axis we get the xticks which are our x label values
categories = ax.get_xticks()
# Putting & Styling lables and title
new_dates = tasks2['Task ID'].dt.strftime('%d/%m/%Y').tolist()  
plt.xticks(categories,new_dates,fontsize=30, rotation=30)
# Adding & Styling lables and title 
plt.xlabel("Task ID",fontsize=100)
plt.ylabel("Task Feature Values",fontsize=100)
plt.title("Distribution of feature values for each Task",fontweight="bold",fontsize=150)
plt.xticks(fontsize=30)
plt.yticks(fontsize=50)
plt.savefig("./Figures/2.1.2_Task_Dist_Box.png")
print("Section 2.1.2: Boxplot Graph is saved in Figures directory")
plt.show() 
print() # Line Gap


''' Interactive Boxplot  '''

print("Interactive Boxplot on localhost")

# Plotting the interactive graph
import plotly.express as px

tasks2_columns=tasks2.columns[1:]
fig1_plotly = px.box(tasks2, y=tasks2_columns,hover_data=['Task ID'],template='seaborn', \
                     title='Distribution of Tasks according to features', \
                     labels={'value':'','variable':'Features'})
fig1_plotly.show()
print("Interactive Plot opens in web browser: http://127.0.0.1:xxxxx/ Close it to move forward! ")
print() # Line Gap


''' 2.2 Boxplot of the distribution of errors if each supplier is chosen to perform every task '''

print("Section 2.2: Boxplot of the distribution of errors if each supplier is chosen to perform every task")
# Equation 1: 𝐸𝑟𝑟𝑜𝑟(𝑡) = min{ 𝑐(𝑠, 𝑡) | 𝑠 ∈ 𝑆 } − 𝑐(𝑠𝑡′,𝑡)
# Equation 2: 𝑆𝑐𝑜𝑟𝑒 =√((∑ 𝐸𝑟𝑟𝑜𝑟(𝑡)^2)/|𝑇|)   
 
# Finding the minimum cost for each task ID as minimum cost is the best cost provided by a supplier
new_costs = costs.groupby('Task ID')['Cost'].min().reset_index()
# Creating a list of all suppliers
supplier_lis = costs['Supplier ID'].unique()
# We are getting exactly 63 values indicating 63 suppliers 
len(supplier_lis)
#Calculate the error for every Task ID + Supplier ID combination
error_out = pd.DataFrame({'Task ID' : [], 'Min Cost' : [], 'Supplier' : [], \
                          'Supplier Cost' : [], 'Error' : []})
for tasks in new_costs.itertuples():
    for supplier in supplier_lis:
        taskname = tasks[1]
        mincost = tasks[2]
        supcost = costs[(costs['Supplier ID'] == supplier) & (costs['Task ID'] == taskname)].iloc[0]['Cost']
        errorval = (mincost - supcost)
        error_append= pd.DataFrame([[taskname, mincost, supplier, supcost, errorval]],\
                                   columns=['Task ID', 'Min Cost', 'Supplier', 'Supplier Cost', 'Error'])
        error_out = pd.concat([error_out, error_append])
#Function rmsefunc : Calculate the RMSE for cost of each supplier
#comb: Task ID + Supplier ID combination 
def rmsefunc(comb):
    rmse = np.sqrt(mean_squared_error(comb['Min Cost'], comb['Supplier Cost']))
    return pd.Series(dict(rmse = rmse))
rmseData = error_out.groupby('Supplier').apply(rmsefunc).reset_index()
#  Boxplot showing distribution of errors for each supplier, if chosen
errorgraph = error_out.copy()
errorgraph['Error'] = errorgraph['Error'].abs()
# Styling the plot
plt.figure(figsize=(45,20))
sns.set_style("darkgrid")
# We plot boxplot with box width as 0.75 to make it less congested
err_plot = sns.boxplot(data=errorgraph, x= 'Supplier', y = 'Error' ,width=0.75)
ax = err_plot.axes  # Plotting axes
# From the axis we get the xticks which are our x label values 
categories = ax.get_xticks()
# For annotating each boxplot with RMSE, we run a for loop through each of the xticks
for cat, supplier in enumerate(supplier_lis):
    currentsupplier = supplier
    y = rmseData[rmseData['Supplier'] == supplier].iloc[0]['rmse'] # Setting RMSE based on index
    y = round(y, 3)
    ax.text(
        cat, 
        y, 
        f'{y}', 
        ha='center', 
        va='center', 
        fontweight='bold', 
        size=13,
        color='white',
        bbox=dict(facecolor='#445A64'))
sns.despine(offset=10, trim=True)
plt.xticks(rotation = 45)
# Putting & Styling lables and title 
plt.xlabel("Suppliers", fontsize=25)
plt.ylabel("Error", fontsize=25)
plt.title("Error Plot for each supplier",fontweight="bold", fontsize=50)
plt.rc('axes', titlesize=35)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=20)
plt.savefig("./Figures/2.2_Boxplot.png")
print("Section 2.1.1: Boxplot Graph is saved in Figures directory")
plt.show()
print() # Line Gap


''' 2.3. Heatmap Plot showing cost values of tasks (rows) and suppliers (columns) '''

print("Section 2.3: Heatmap Plot showing cost values of tasks (rows) and suppliers (columns) ")
## As the data frame was initially long format, 
## it is converted to wide format with Task ID as rows and Supplier ID as columns. 
costs_h=costs.pivot_table(values="Cost", index="Task ID", columns="Supplier ID")
print("costs_h table in wide format: ")
print(costs_h.info())

## "natsort" module is used to sort the Supplier ID in order, 
## since Supplier ID consisted of both alphabets and numbers. 
costs_h=costs_h.reindex(ns.natsorted(costs_h.columns), axis=1)

## Seaborn module used to plot heatmap, showing cost of each task quoted by each supplier
pltlib.pyplot.figure(figsize=(80,60))
fig2 = sns.heatmap(costs_h, annot=True, cmap="coolwarm")
ax2 = fig2.axes
cat2 = ax2.get_yticks()
plt.yticks(cat2,new_dates)
plt.savefig("./Figures/2.3.1_Heatmap_Seaborn.png")
print("Section 2.3.1: Heatmap is saved in Figures directory")
plt.show()
print() # Line Gap

# Comparison between suppliers for each Task, Red: Most expensive , Blue: Cheapest
costheat = costs_h.style.format("{:.2}")\
           .background_gradient(cmap='coolwarm',axis=1).format_index('{:%d/%m/%Y}')

print(costheat)

import dataframe_image as dfi  

dfi.export(costheat, 'Figures/2.3.2_Heatmap_Dataframe.png', max_cols=-1, max_rows=-1)
print("Section 2.3.2: Customised Heatmap is saved in Figures directory")
print() # Line Gap

error_best = error_out[error_out['Supplier']=='S56']
error_best.reset_index(inplace = True)

from math import sqrt
rmse_best = sqrt(np.mean((error_best['Error'])**2))