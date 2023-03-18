The following steps are taken for writing the credit card fraud detection software
* Imported necessary libraries 
* Identify the path to the dataset and import the dataset. Currently, the dataset is located in the same directory that fraudDetection.py file is located.
* Transform the dataset to a data frame so it's easier to locate information.
* Classify non-fraud transactions as 0 and fraud as 1. 
* Split data by variables of class and amount.
* Train the Decision Tree and Random Forest classifiers.
* Save the trained model in a file so that running the software script doesn't train it every time it runs. 
* The result of each classifier is evaluated during the training phase so that the best one can be selected based on its precision. 
* The accuracy of each classifier is detected, and the higher-accuracy algorithm is allowed to print the final result. 
