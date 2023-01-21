import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

''' Importing the declared variables from previous steps'''
from Step3_ModelFitting import trains, x, y, lasso, x_train, y_train, ridge, Lasso, Ridge, ElasticNet, x_test, y_test, Elnet
from Step4_RMSE import rmse_lasso, rmse_ridge, rmse_elastic, Test_data3
# only required variables are imported to reduce Ram redundancy

''' 4. Leave One Out Group Cross Validation'''
print("Section 4: Leave One Out Group Cross Validation")


from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import make_scorer

logo=LeaveOneGroupOut() # using leave-one-out-group cross validation
train_Groups=trains['Task ID']
logo.get_n_splits(x,y,train_Groups)
logo.get_n_splits(groups=train_Groups)
print("Length of Train Group: ", len(train_Groups))
print() # Line Gap

#check number of cores
n_cpu=os.cpu_count()
print("Number of CPU count: ", n_cpu)
print() # Line Gap

def error_2(y_true,y_pred):
    selected=y_true.iloc[np.argmin(y_pred)] 
    #We select the real cost of the supplier recommended by the model (the cheapest one)
    minimal=np.min(y_true)  
    # We select the minimal value for each task
    error=minimal-selected 
    #Finally we applied equation1 (the minimal cost - the cost if we select what the model suggest)
    return(error)

''' 4.1. Lasso Model '''

print("Section 4.1: Lasso Model :")

BeePy_scorer = make_scorer(error_2) # making our own scorer
scores_lasso = cross_val_score(lasso, x_train, y_train, scoring = BeePy_scorer, \
                               cv=logo, groups=train_Groups) #takes the features df and target y , splits into k-folds
#Or we can just set as -1 : -1 is for using all the CPU cores available.

print("Lasso Scores: ", scores_lasso)

#rmse calculation
error=np.power(scores_lasso,2) # squaring score values
mse=np.sum(error)/len(scores_lasso) # calculating mean of squared score values
rmse4_lasso=np.sqrt(mse) # finding square root to get RMSE for Lasso
print("RMSE Lasso :", rmse4_lasso)

# RMSE comparison for Lasso model
rmse_lasso['Cross Val']=rmse4_lasso
print("RMSE comparison for Lasso: " )
print(rmse_lasso) # RMSE of scores returned through Lasso
print() # Line Gap


''' 4.2 Ridge Model '''

print("Section 4.2: Ridge Model :")

scores_ridge = cross_val_score(ridge, x_train, y_train, \
                               scoring = BeePy_scorer, cv=logo, \
                               groups=train_Groups, n_jobs=10)
print("Ridge Scores :", scores_ridge)


quadraticerror=np.sum(np.power(scores_ridge,2)) # squaring score values
quadraticerror_2=quadraticerror/len(scores_ridge) # calculating mean of squared score values
rmse4_ridge=np.sqrt(quadraticerror_2) # finding square root to get RMSE for Ridge
print("RMSE Ridge :", rmse4_ridge)

# RMSE comparison for Ridge model
rmse_ridge['Cross Val']=rmse4_ridge
print("RMSE comparison for Ridge: ")
print(rmse_ridge) # RMSE of scores returned through Ridge
print() # Line Gap


''' 4.3 ElasticNet Model '''

print("Section 4.3: ElasticNet Model :")

scores_elastic = cross_val_score(Elnet, x_train, y_train, \
                               scoring = BeePy_scorer, cv=logo, \
                               groups=train_Groups, n_jobs=-1)
print("Elastic Scores: ", scores_elastic)


quadraticerror=np.sum(np.power(scores_elastic,2)) # squaring score values
quadraticerror_2=quadraticerror/len(scores_elastic) # calculating mean of squared score values
rmse4_elastic=np.sqrt(quadraticerror_2) # finding square root to get RMSE for Elastic
print("RMSE Elastic :",  rmse4_elastic)

# RMSE comparison for ElasticNet model
rmse_elastic['Cross Val']=rmse4_elastic
print("RMSE comparison for ElasticNet: ")
print(rmse_elastic) # RMSE of scores returned through Elastic
print() # Line Gap


'''    5. Hyper-Parameter Optimization      '''

print("Section 5: Hyper-Parameter Optimization")

# importing GridSearchCV for this section
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# Importing & Using GridSearchCV to find best hyper parameter

print("Section 5.1: Lasso Model :")

# Trying different range of Alpha values & narrowing it
#param_grid = {"alpha":[1e-8, 1e-7, 1e-6, 4.286353962275511e-05,1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]}
alphas = np.linspace (0.1, 0.0001, 100)
param_grid={"alpha": alphas}

grid_search = GridSearchCV(lasso, param_grid, scoring=BeePy_scorer, cv=logo, n_jobs = -1)
grid_search.fit(x_train, y_train, groups=train_Groups)

print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
grid_search.score(x_train, y_train)
gsTable = pd.DataFrame(grid_search.cv_results_)
print("GridSearch Table:")
print(gsTable)
print(gsTable[['params', 'mean_test_score', 'std_test_score',	'rank_test_score']])

#Lasso: Alpha optimization in graph
fig1=sns.lineplot(data=gsTable[['param_alpha', 'mean_test_score', 'std_test_score',	'rank_test_score']],x='param_alpha',y='mean_test_score',markers= True)
fig1.set(xscale='log')
plt.xticks(rotation=45)
fig1.set(xlabel='Alpha (1)', ylabel='Error (Equation1)')
fig1.axes.set_title("Hyper Parameter Optimization - Lasso",fontsize=18)
plt.savefig("./Figures/5.1_Lasso_Optimization.png")
plt.show()
print("Lasso Alpha Optimization graph is saved in Figures directory")

alpha_param=grid_search.best_params_['alpha']
# Training the model with the best hyper parameters we got from grid search and scoring it ​
lassoHyper = Lasso(alpha = alpha_param, max_iter=10000, random_state=42)
lassoHyper.fit(x_train, y_train)
pred_lassoHyper = lassoHyper.predict(x_test)
print(pred_lassoHyper)
print("Lasso Score after Hyper-Parameter Optimization: ", lassoHyper.score(x_test,y_test))

# Getting RMSE from the predicted values from hyper parameters and comparing it ​
scores_lasso = cross_val_score(lassoHyper, x_train, y_train, scoring = BeePy_scorer, \
    cv=logo, groups=train_Groups, n_jobs=-1)

error=np.power(scores_lasso,2)
mse=np.sum(error)/len(scores_lasso)
rmse5_lasso=np.sqrt(mse)
rmse_lasso['Hyper-parameter']=rmse5_lasso
print(rmse_lasso) # RMSE comparison with different section
print() # Line Gap


''' 5.2 Ridge Model '''

print("Section 5.2: Ridge Model :")


# param_grid = {"alpha":[1e-8, 1e-7, 1e-6, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]}
alphas = np.linspace (1e-8, 10000, 1000)
param_grid={"alpha": alphas}

grid_search2 = GridSearchCV(ridge, param_grid, scoring=BeePy_scorer, cv=logo, n_jobs = -1)
grid_search2.fit(x_train, y_train, groups=train_Groups)
print("Best Parameters: ",grid_search2.best_params_)
print("Best Score: ", grid_search2.best_score_)
print("Ridge Score after Hyper-Parameter Optimization: ", grid_search2.score(x_train, y_train))
gsTable2 = pd.DataFrame(grid_search2.cv_results_)
print(gsTable2)

#Ridge: Alpha optimization in graph
fig2=sns.lineplot(data=gsTable2[['param_alpha', 'mean_test_score', 'std_test_score','rank_test_score']],x='param_alpha',y='mean_test_score',markers= True)
fig2.set(xscale='log')
plt.xticks(rotation=45)
fig2.set(xlabel='Alpha (1)', ylabel='Error (Equation1)')
fig2.axes.set_title("Hyper Parameter Optimization - Ridge",fontsize=18)
plt.savefig("./Figures/5.2_Ridge_Optimization.png")
plt.show()
print("Ridge Alpha Optimization graph is saved in Figures directory")

alpha_param=grid_search2.best_params_['alpha']
# Training the model with the best hyper parameters we got from grid search and scoring it 
ridgeHyper = Ridge(alpha = alpha_param, max_iter=10000, random_state=42)
print("Ridge Score after Hyper-Parameter Optimization: ", ridgeHyper.fit(x_train, y_train))
pred_ridgeHyper = ridgeHyper.predict(x_test)
print(pred_ridgeHyper)
print("Ridge Hyper Score: ",ridgeHyper.score(x_test,y_test)) # Ridge score

#Exploration: Why we are getting the same Error Score for all the alphas used. 

alphas = np.linspace (1e-8, 2000, 1000)
test_score=[]

for alpha in alphas:
    ridgeHyper = Ridge(alpha =alpha, max_iter=10000, random_state=42)
    ridgeHyper.fit(x_train, y_train)
    
    pred_ridgeHyper = ridgeHyper.predict(x_test)
    pred_ridgeHyper

    test_score.append(ridgeHyper.score(x_test,y_test))

#Conversion to dataframe to explain
data_ridge = pd.DataFrame(
    {'alphas': alphas,
     'test_score': test_score,
    })

r2_ridge=sns.lineplot(data=data_ridge,x='alphas',y='test_score')
r2_ridge.set_ylim(0.4, 0.7)
r2_ridge.set(xlabel='Alpha', ylabel='R2')
r2_ridge.axes.set_title("Changes on Accuracy",fontsize=18)
plt.savefig("./Figures/5_Ridge_R2.png")
print("Ridge Change of Accuracy graph is printed")
plt.show()

#No matter what alpha we use from a range from 1e-8 to 100
#The capacity of the model to predict the variance of the Cost (Accuracy - R2) changes slightly (less than 0.10).
#Therefore, the suppliers predicted for the model do not change. Consequently, the rest applied by the Equation 1 remains the same (minor price - actual cost of the supplier recommended by the model)

# Getting RMSE from the predicted values from hyper parameters and comparing it
scores_ridge = cross_val_score(ridgeHyper, x_train, y_train, scoring = BeePy_scorer, \
    cv=logo, groups=train_Groups, n_jobs=-1)

error=np.power(scores_ridge,2)
mse=np.sum(error)/len(scores_ridge)
rmse5_ridge=np.sqrt(mse)
rmse_ridge['Hyper-parameter']=rmse5_ridge
print(rmse_ridge) # RMSE comparison for different models
print() # Line Gap


''' 5.3 ElasticNet Model'''

print("Section 5.3: ElasticNet Model :")

#param_grid = {"alpha":[1e-5,1e-4,1e-3,1e-2, 1, 10], "l1_ratio":[1e-5,1e-4, 1e-3,1e-2, 1]} # First Range
param_grid = {"alpha":[0.01,0.1,1.0,10,100],"l1_ratio":np.linspace(0.1,0.001,100)} # Second Range

gs3 = GridSearchCV(Elnet, param_grid, scoring=BeePy_scorer, cv=logo, n_jobs = -1)
gs3.fit(x_train, y_train, groups=train_Groups)
print("Best Parameters: ",gs3.best_params_)
print("Best Score: ", gs3.best_score_)
print("Elastic Score after Hyper-Parameter Optimization: ", gs3.score(x_train, y_train))
gsTable3 = pd.DataFrame(gs3.cv_results_)
print(gsTable3[['params','mean_test_score','rank_test_score']])

gsTable3_plot=pd.DataFrame(gsTable3[['params','param_alpha','param_l1_ratio','mean_test_score','rank_test_score']])
gsTable3_plot=gsTable3_plot.loc[(gsTable3_plot['param_alpha'].isin([0.01, 0.1, 1.0, 10, 100]))&(gsTable3_plot['param_l1_ratio'].isin([0.1,0.05,0.023000000000000007,0.01200000000000001]))]

#Elastic: Alpha optimization in graph
fig3=sns.lineplot(data=gsTable3_plot,x='param_alpha',y='mean_test_score',hue='param_l1_ratio',markers= True,palette=['blue','green','black','yellow'])
fig3.set(xscale='log')
plt.xticks(rotation=45)
fig3.set(xlabel='Alpha (1)', ylabel='Error (Equation1)')
fig3.axes.set_title("Hyper Parameter Optimization - Elastic",fontsize=18)
plt.savefig("./Figures/5.3_Elastic_Optimization.png")
plt.show()
print("ElasticNet Alpha Optimization graph is saved in Figures directory")

Elnet = ElasticNet(alpha= gs3.best_params_['alpha'], l1_ratio= gs3.best_params_['l1_ratio'], max_iter=10000, random_state=42)
Elnet.fit(x_train, y_train)
pred_ElasticNet = Elnet.predict(x_test)
print(pred_ElasticNet)
print("ElasticNet Score: ",Elnet.score(x_test,y_test)) # ElasticNet score

# Getting RMSE from the predicted values from hyper parameters and comparing it
scores_elastic = cross_val_score(Elnet, x_train, y_train, scoring = BeePy_scorer, \
    cv=logo, groups=train_Groups, n_jobs=-1)

error=np.power(scores_elastic,2)
mse=np.sum(error)/len(scores_elastic)
rmse5_ela=np.sqrt(mse)

rmse_elastic['Hyper-parameter']=rmse5_ela


print(rmse_elastic) # RMSE comparison for different models
print() # Line Gap

# Concatinating different models
total_model = pd.concat([rmse_lasso,rmse_ridge,rmse_elastic],axis = 0)
total_model.insert(0,'Model',['Lasso','Ridge','Elastic'])
total_model = total_model.set_index('Model') # Reseting index as model
print('Summary of RMSE scores throughout Machine Learning Development Process')
print(total_model)
print() # Line space

# Total
print("Line graph representing RMSE for Section 2.2 and Different Models")

ax = total_model.T.plot(figsize = (7,6),marker = 'o')
ax.set_ylabel('RMSE',fontsize = 12)
ax.set_xlabel('Section',fontsize = 12)
#Fist Approach
plt.text(0.07, 0.026, str(round(total_model.loc['Lasso','Manual approach (2.2)'],5)), size='medium', color='black', weight='semibold')
#Lasso
plt.text(0.63, 0.05, str(round(total_model.loc['Lasso','Initial ML'],5)), size='medium', color='black', weight='semibold')
plt.text(1.63, 0.0579, str(round(total_model.loc['Lasso','Cross Val'],5)), size='medium', color='black', weight='semibold')
plt.text(3, 0.033, 'Lasso:' + str(round(total_model.loc['Lasso','Hyper-parameter'],5)), size='medium', color='black', weight='semibold')
#Ridge
plt.text(0.69, 0.0415, str(round(total_model.loc['Ridge','Initial ML'],5)), size='medium', color='black', weight='semibold')
plt.text(1.69, 0.038, str(round(total_model.loc['Ridge','Cross Val'],5)), size='medium', color='black', weight='semibold')
plt.text(3, 0.027, 'Ridge:' + str(round(total_model.loc['Ridge','Hyper-parameter'],5)), size='medium', color='black', weight='semibold')
#Elastic
plt.text(0.69, 0.0385, str(round(total_model.loc['Elastic','Initial ML'],5)), size='medium', color='black', weight='semibold')
plt.text(1.69, 0.043, str(round(total_model.loc['Elastic','Cross Val'],5)), size='medium', color='black', weight='semibold')
plt.text(3, 0.030, 'Elastic:' + str(round(total_model.loc['Elastic','Hyper-parameter'],5)), size='medium', color='black', weight='semibold')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 12)
plt.savefig("./Figures/Section_RMSE_Compare.png")
print("Section 5.3: RMSE Comparison Graph saved in Figures directory")
plt.show()
print()

print("Thank you <3")
print("Group - BeePy")
print("""

                __/   _
        .__  __.  \__/   __
         .-`'-.   /  \__/
     .-.(  oo  ).-. _/  \__/
 __ :   \".~~."/   ; \__/
/  \_`.  Y`--'Y  .' _/  \__
 __/  `./======\.'   \__/  \_
/  \__/ \======/  \__/  \__/
\__/   (_`----'_)    \__/  \ (BeePy)

""")