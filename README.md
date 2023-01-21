Installation & Running

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

- Group BeePy


