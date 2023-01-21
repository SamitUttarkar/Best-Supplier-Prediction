

![🖥_Predicting_Best_Supplier_for_Acme_Corporation](https://user-images.githubusercontent.com/67644483/213885099-a51caad6-8472-4ff0-9bd0-c56734c7318b.png)

# **Prpject Breif**

**This projects demonstrates the use of machine learning (ML) models in recommending the best suppliers to Acme Corporation, given a set of task features. This process begins with data cleaning and preparation, exploratory data analysis (EDA) and machine learning, where results from predicted data can be obtained. Cross-validation (Leave-One-Group-Out) is used to validate the ML model score, and hyper-parameter optimization is further used to find the optimal set of hyper-parameters for improved predictions.**

## Data Coding and Analysis

Three datasets were provided for this project:
- [tasks.xlsx](https://github.com/SamitUttarkar/Best-Supplier-Prediction/blob/main/Support_Data/cost.csv)
- costs.csv
- suppliers.csv

The attributes for all three datasets were then obtained. Table 1 (before) describes the obtained attributes.

### Data Preparation 

To obtain suitable datasets for exploratory data analysis and machine learning algorithms, the data preparation phase was carried out prior to the main data analysis process.
The data cleaning consisted of 5 tasks with different objectives.
1. Checking for missing values and the proper format of variables according to their nature (e.g., Cost as floats).
2. Aligning the format of the key variables in all databases (Task ID/ Supplier ID) to identify any task that does not have a related cost (dependent variable in future models) and to merge the datasets without conflict.
3. Variables (task features) with low variability, indicated by a variance of less than 0.01, were eliminated to keep only attributes heterogeneous enough to provide relevant insights for exploration and modelling. 35 Task Features were dropped.
4. Variables (all features) were standardized on a scale of -1 to 1 to facilitate comparisons between features in EDA and model training.
5. Highly correlated variables were filtered out to avoid multicollinearity and redundancy problems in subsequent models. 54 task features that exhibited 0.8 or more correlation with other task features were dropped.
6. Filtering out low-performance suppliers indicated by the inability to appear in the top 20 costs for any task. Only one supplier did not meet this requirement, hence, was dropped.

![Screenshot 2023-01-21 at 8 13 32 PM](https://user-images.githubusercontent.com/67644483/213885468-0135a47e-9e94-4710-b00f-645b00b36895.png)


 ## Installation & Running

- Download Zip Folder BeePy_Python (Unzip)

- Files Structure

          2 Folders & 5 .py files
        - Support_Data
            - tasks.xlsx
            - cost.csv
            - suppliers.csv

        - Figures (Automatically gets created after execution of Acme Python)
            - SectionNo_Image.png (Starts saving after succesful exceution of Acme Python)

        - Step1_DataPreprocessing.py
        - Step2_EDA.py
        - Step3_ModelFitting.py
        - Step4_RMSE.py
        - Step5_CV_&_HPO.py (Cross Validation and Hyper-parameter Optimization)
        - main.py
        
        - BeePy Presentation.pdf
        - BeePy.ipynb
    
- Dependencies required / Installation

        - Open terminal/cmd/Code Editor(Spyder/ PyCharm)
        - python 3.7 + (respective pip version) / stable conda
        - standard packages like numpy, pandas are added in top of the .py files
        - section specific packages are included in different section
        - all those packages are necessary to run the code smoothly
                -  Method 1 )   pip/conda install package name is needed to 
                                - Dependencies needed :
                                - pandas, numpy, copy, 
                                - os, dataframe_image, sklearn, 
                                - matplotlib, seaborn, natsort, 
                                - plotly, random, etc.
                                - Some dependencies/packages require specific versions of python, pip, pandas, numpy combination,
                          hence, kindly use the stable versions
        - requirements.txt has the versions included for different packages
        - In Terminal/Cmd run > pip/conda install -r requirements.txt

- Running Python File

        - Move to respective directory in cmd/terminal/code editor 
          ex - cd "Downloads/python_BeePy/Py_Files" 

        - In terminal/cmd run > python Step5_CV_&_HPO.py
        - Or run >python main.py
        - Step 5 includes all the previous steps imported
        - Individual steps can also run individually




