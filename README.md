The following steps are taken for writing the credit card fraud detection software
* Imported necessary libraries 
* Identified the path to the dataset and import the dataset. Currently, the dataset is located in the same directory that fraudDetection.py file is located. The dataset was too large to include in the repository. A copy of the dataset can be found at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
* Transformed the dataset to a data frame so it's easier to locate information.
* Classified non-fraud transactions as 0 and fraud as 1. 
* Split data by variables of class and amount.
* Trained the Decision Tree and Random Forest classifiers.
* Saved the trained model in a file so that running the software script doesn't train it every time it runs. 
* The first time the fraudDetection.py file is run it gets trained and the trained model is saved in two seperate files; one for Random Forest and one for Decision Tree. The second time the python scrift is run it will use the saved models in the files for detecting fraud.
* The result of each classifier is evaluated during the training phase so that the best one can be selected based on its precision. 
* The accuracy of each classifier is detected, and the higher-accuracy algorithm is allowed to print the final result.
* The following is an example of how the output of fraudDetection.py should look like as of March 18, 2023
Number of Genuine transactions:  284315
Number of Fraud transactions:  492
Percentage of Fraud transactions: 0.1727
Shape of train_input_attributes:  (199364, 29)
Shape of test_input_attributes:  (85443, 29)
Random Forest Score:  99.95786664794073
Decision Tree Score:  99.93328885923948
Confusion Matrix - Decision Tree
[[85276    31]
 [   26   110]]
Confusion matrix, without normalization
Confusion Matrix - Random Forest
[[85298     9]
 [   27   109]]
Confusion matrix, without normalization
Evaluation of Decision Tree Model
Accuracy: 0.99933
Precision: 0.78014
Recall: 0.80882
F1-score: 0.79422
Evaluation of Random Forest Model
Accuracy: 0.99958
Precision: 0.92373
Recall: 0.80147
F1-score: 0.85827
