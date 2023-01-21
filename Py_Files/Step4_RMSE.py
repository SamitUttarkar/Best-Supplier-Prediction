import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns


''' Importing the declared variables from previous steps'''
from Step2_EDA import error_out, rmseData, error_best, rmse_best
from Step3_ModelFitting import tests, pred_lasso, pred_ridge, pred_ElasticNet
# Required variables from Step 2 & Step 3 are imported

''' 3.4 Calculating RMSE values'''

print("Section 3.4: Calculating RMSE values:")
print() # Line Gap

# Calculating the Error of the trained model for each task in TestGroup and using Equation 1 


''' 3.4.1 Lasso Model '''

print("Section 3.4.1: Lasso Model")
Test_data=copy.deepcopy(tests.iloc[:, [0,1,47]]) #copy the data of test dataset
Test_data['Pred_cost']=pred_lasso 

print("Unique values in Test Data Lasso Model['Task ID']: ", len(Test_data['Task ID'].unique()))

Test_data=Test_data.set_index('Task ID')
Test_data['Min_cost']=Test_data.groupby("Task ID").Cost.agg(min=np.min)  #Min cost by task
print("Test Data for Lasso Model:")
print(Test_data)

#Sort the values by Task ID and Pred_cost in ascending order to get the cheapest pred_cost 
#in order to find the best supplier predicted by machine learning by Task ID

Test_data3=Test_data.sort_values(by=['Task ID','Pred_cost'], ascending=[True, True])
Test_data3=Test_data3.reset_index()
Test_data3['Error']=Test_data3['Min_cost']-Test_data3['Cost']
Test_data3=Test_data3.reset_index()


#Variables to get the first row of each Task ID group
Supplier_Q=len(Test_data3['Supplier ID'].unique())
length_data=len(Test_data3['Task ID'])

Test_data4=Test_data3[0:length_data:Supplier_Q]
Test_data4=Test_data4.iloc[:,1:]
print(Test_data4)


# (2) Calculate the RMSE score

Test_data4['Error_Squared']=Test_data4['Error']**2
mse=np.sum(Test_data4['Error_Squared'])/len(Test_data4['Task ID'].unique())

rmse3_lasso=np.sqrt(mse)
print("RMSE value for Lasso Model: ")
print(rmse3_lasso)
print() # Line Gap


''' 3.4.2 Ridge model  '''

print("Section 3.4.2: Ridge model :")

# (1) Calculate the Error of the trained model for each task in TestGroup and using Equation 1 

Test_dataRidge=copy.deepcopy(tests.iloc[:, [0,1,47]]) #copy the data of test dataset
Test_dataRidge['Pred_cost']=pred_ridge #include the prediction result 
print("Unique values in Test Data Ridge Model['Task ID']: ", len(Test_dataRidge['Task ID'].unique()))

Test_dataRidge=Test_dataRidge.set_index('Task ID')
Test_dataRidge['Min_cost']=Test_dataRidge.groupby("Task ID").Cost.agg(min=np.min)  #Min cost by task
print("Test Data for Ridge Model:")
print(Test_dataRidge)

Test_dataRidge3=Test_dataRidge.sort_values(by=['Task ID','Pred_cost'], ascending=[True, True])
Test_dataRidge3=Test_dataRidge3.reset_index()

Test_dataRidge3['Error']=Test_dataRidge3['Min_cost']-Test_dataRidge3['Cost']
Test_dataRidge3=Test_dataRidge3.reset_index()

#Sort the values by Task ID and Pred_cost in ascending order to get the cheapest pred_cost 
#in order to find the best supplier predicted by machine learning by Task ID

Supplier_QRidge=len(Test_dataRidge3['Supplier ID'].unique())
length_data=len(Test_dataRidge3['Task ID'])

Test_dataRidge4=Test_dataRidge3[0:length_data:Supplier_QRidge]
Test_dataRidge4=Test_dataRidge4.iloc[:,1:]
print(Test_dataRidge4)

# (2) Calculate the RMSE score 

Test_dataRidge4['Error_Squared']=Test_dataRidge4['Error']**2
mse=np.sum(Test_dataRidge4['Error_Squared'])/len(Test_dataRidge4['Task ID'].unique())

rmse3_ridge=np.sqrt(mse)
print("RMSE value for Ridge Model: ")
print(rmse3_ridge)
print() # Line Gap


''' 3.4.3 ElasticNet Model'''

print("Section 3.4.3: ElasticNet Model : ")

Test_dataElasticNet=copy.deepcopy(tests.iloc[:, [0,1,47]]) #copy the data of test dataset
Test_dataElasticNet['Pred_cost']=pred_ElasticNet #include the prediction result 
len(Test_dataElasticNet['Task ID'].unique())


Test_dataElasticNet=Test_dataElasticNet.set_index('Task ID')
Test_dataElasticNet['Min_cost']=Test_dataElasticNet.groupby("Task ID").Cost.agg(min=np.min)  #Min cost by task
print("Test Data for Lasso Model:")
print(Test_dataElasticNet)

Test_dataElasticNet3=Test_dataElasticNet.sort_values(by=['Task ID','Pred_cost'], ascending=[True, True])
Test_dataElasticNet3=Test_dataElasticNet3.reset_index()

Test_dataElasticNet3['Error']=Test_dataElasticNet3['Min_cost']-Test_dataElasticNet3['Cost']
Test_dataElasticNet3=Test_dataElasticNet3.reset_index()

#Sort the values by Task ID and Pred_cost in ascending order to get the cheapest pred_cost 
#in order to find the best supplier predicted by machine learning by Task ID

Supplier_QRidge=len(Test_dataElasticNet3['Supplier ID'].unique())
length_data=len(Test_dataElasticNet3['Task ID'])

Test_dataElasticNet4=Test_dataElasticNet3[0:length_data:Supplier_QRidge]
Test_dataElasticNet4=Test_dataElasticNet4.iloc[:,1:]
print(Test_dataElasticNet4)

# (2) Calculate the RMSE score

Test_dataElasticNet4['Error_Squared']=Test_dataElasticNet4['Error']**2
mse=np.sum(Test_dataElasticNet4['Error_Squared'])/len(Test_dataElasticNet4['Task ID'].unique())

rmse3_ElasticNet=np.sqrt(mse)
print("RMSE value for ElasticNet Model: ")
print(rmse3_ElasticNet)
print() # Line Gap


''' 3.5 Comparison of Error and RMSE '''


print("Section 3.5: Comparison of Error and RMSE")

# Comparison between error values obtained from section 2.2 and section 3.4 (predictive models)

error_out.reset_index(inplace=True)  ## Dataframe obtained from Section 2.2
error_out = error_out.drop(['index'], axis=1)

"""
The function print_error_2_2 fetches the error obtained in Section 2.2 
based on Task ID and Supplier in the predictive model.
Hence, it helps in comparing the Error values obtained manually versus the predictive model. 

"""

def print_error_2_2(df):  

    Error_22 = []

    for i in df.index:
        for j in range(len(error_out)):
            if (df['Task ID'][i] == error_out['Task ID'][j] ) \
            and (df['Supplier ID'][i] == error_out['Supplier'][j]):
                Error_22.append(error_out['Error'][j])

    #print(Error_22)
    
    df['Error 2.2'] = Error_22

print(Test_data4)

print_error_2_2(Test_dataRidge4) ## Adding corresponding Error value from Section 2.2 
                                ## into the dataframe obtained after Ridge Regression
print(Test_dataRidge4)

sp_lasso=Test_data4['Supplier ID']
t_id_lasso= Test_data4['Task ID']

sp_ridge=Test_dataRidge4['Supplier ID']
t_id_ridge= Test_dataRidge4['Task ID']

rmseData.set_index('Supplier', inplace=True)
rmseData_t=rmseData.T

print("rmseData :", rmseData_t)

r_col=rmseData_t.columns

index_lasso=r_col[r_col.isin(sp_lasso)]
index_ridge=r_col[r_col.isin(sp_ridge)]

# rmse comparison dataframe

# Lasso
rmse_lasso=pd.DataFrame(np.array([rmse_best]), columns=['Manual approach (2.2)'])
rmse_lasso['Initial ML']=rmse3_lasso
print("RMSE comparison for Lasso Model: ")
print(rmse_lasso)
print() # Line Gap

# Ridge
rmse_ridge=pd.DataFrame(np.array([rmse_best]), columns=['Manual approach (2.2)'])
rmse_ridge['Initial ML']=rmse3_ridge
print("RMSE comparison for Ridge Model: ")
print(rmse_ridge)
print() # Line Gap

# ElasticNet
rmse_elastic=pd.DataFrame(np.array([rmse_best]), columns=['Manual approach (2.2)'])
rmse_elastic['Initial ML']=rmse3_ElasticNet
print("RMSE comparison for ElasticNet Model: ")
print(rmse_elastic)
print() # Line Gap


print("Section 3.5: RMSE & Error comparison boxplot ")
# Set the figure size
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.figure(figsize=(7,5))

data1 = pd.DataFrame({"2.2":abs(error_best['Error']) ,  "Lasso": abs(Test_data4['Error']), "Ridge": abs(Test_dataRidge4['Error']),"Elastic": abs(Test_dataElasticNet4['Error'])})# Plot the dataframe
#ax = data[['Box1', 'Box2']].plot(kind='box', title='boxplot',figsize=(10, 6))
ax = sns.boxplot(data = data1)
ax1 = ax.axes
categories = ax1.get_xticks()
rmse_lis = [rmse_best,rmse3_lasso,rmse3_ridge,rmse3_ElasticNet]
for cat in categories:
    y = rmse_lis[cat]
    y = round(y,4)
    ax.text(
    cat,
     y,
     f'{y}',
     ha='center',
     va='center_baseline',
     fontweight='bold',
     size=13,
     color='white',
     bbox=dict(facecolor='#445A64'))
sns.despine(offset=10, trim=True)
plt.xticks(rotation = 45)
# Putting & Styling lables and title 
plt.xlabel("Suppliers", fontsize=10)
plt.ylabel("Error", fontsize=10)
plt.title("Error Plot for each supplier",fontweight="bold", fontsize=10)
plt.rc('axes', titlesize=35)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=20)
plt.savefig("./Figures/3.5_RMSE_Error_Comparison.png")
print("Section 3.5: RMSE & Error comparison boxplot is saved in Figures directory ")
plt.show()




