The following steps are taken for writing the credit card fraud detection software
* Import necessary libraries 
* Identify the path to the dataset and import the dataset. Currently, the dataset is located in the same directory that fraudDetection.py file is located. The dataset was too large to include in the repository. A copy of the dataset can be found at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
* Transform the dataset to a data frame so it's easier to locate information.
* Classify non-fraud transactions as 0 and fraud as 1. 
* Split data by variables of class and amount.
* Train the Decision Tree and Random Forest classifiers.
* Save the trained model in a file so that running the software script doesn't train it every time it runs. 
* The first time the fraudDetection.py file is run it gets trained and the trained model is saved in two seperate files; one for Random Forest and one for Decision Tree. The second time the python scrift is run it will use the saved models in the files for detecting fraud.
* The result of each classifier is evaluated during the training phase so that the best one can be selected based on its precision. 
* The accuracy of each classifier is detected, and the higher-accuracy algorithm is allowed to print the final result.
* The following lines are an example of how the output of fraudDetection.py should look.
<br />    Number of Genuine transactions:  284315
<br />    Number of Fraud transactions:  492
<br />    Percentage of Fraud transactions: 0.1727
<br />    Shape of train_input_attributes:  (199364, 29)
<br />    Shape of test_input_attributes:  (85443, 29)
<br />    Random Forest Score:  99.95786664794073
<br />    Decision Tree Score:  99.93328885923948
<br />    Confusion Matrix - Decision Tree
<br />    [[85276    31]
<br />     [   26   110]]
<br />    Confusion matrix, without normalization
<br />    Confusion Matrix - Random Forest
<br />    [[85298     9]
<br />     [   27   109]]
<br />    Confusion matrix, without normalization
<br />    Evaluation of Decision Tree Model
<br />    Accuracy: 0.99933
<br />    Precision: 0.78014
<br />    Recall: 0.80882
<br />    F1-score: 0.79422
<br />    Evaluation of Random Forest Model
<br />    Accuracy: 0.99958
<br />    Precision: 0.92373
<br />    Recall: 0.80147
<br />    F1-score: 0.85827
