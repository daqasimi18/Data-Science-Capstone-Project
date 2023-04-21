#https://data-flair.training/blogs/credit-card-fraud-detection-python-machine-learning/
# Necessary libraries and modules
import numpy as np
import pandas as pd
import os.path
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

# The dataset used for this project is in the same directory as the .py file
datastructure = pd.read_csv("./creditcard.csv")
# Check for any null values in the dataset
datastructure.isnull().values.any()
#datastructure["Amount"].describe()

# In the dataset if class is 0 it's none fraud
# If the class column is 1, it represents fraud
non_fraud = len(datastructure[datastructure.Class == 0])
fraud = len(datastructure[datastructure.Class == 1])
# Percentage of fraud from 284,807 transactions
fraud_percent = (fraud / (fraud + non_fraud)) * 100
print("Number of genuine transactions: ", non_fraud)
print("Number of fraud transactions: ", fraud)
print("Percentage of Fraud transactions: {:.4f}".format(fraud_percent))

# Scale the data in the dataset. Drop the Amount and Time variables. 
# Since about 99% of transaction are genuine, the actual Amount
# column in the dataset is dropped for scaling data. Instead a new 
# Amount column is added with scaled values 
scaler = StandardScaler()
datastructure["NormalizedAmount"] = scaler.fit_transform(datastructure["Amount"].values.reshape(-1, 1))
datastructure.drop(["Amount", "Time"], inplace= True, axis= 1)
output_attributes = datastructure["Class"]
input_attributes = datastructure.drop(["Class"], axis= 1)
output_attributes.head()

# The original dataset is divided in two parts, seventy percent of it is used 
# for training the model and thirty percent is used for validation.
(train_input_attributes, test_input_attributes, train_output_attributes, test_output_attributes) = train_test_split(input_attributes, output_attributes, test_size= 0.3, random_state= 42)
print("Shape of train_input_attributes: ", train_input_attributes.shape)
print("Shape of test_input_attributes: ", test_input_attributes.shape)

# Use of multiple models on the dataset and observing which one produces the
# best result e.g. Decision Tree and Random Forest. At the end only one modle 
# will be used for detecting fraud. The first model is DecisionTreeClassifier 
# In the following eight lines of code the decision tree and random forest model
# are trained using the fit() function and recorded the predictions of the 
# model using predict() fuction. Each model may have different scores.
# Decision Tree
if os.path.isfile("./decision_tree_model") == True:
    decision_tree = DecisionTreeClassifier()
    with open("decision_tree_model", "rb") as dt:
        decision_tree_mod = pickle.load(dt)
    decision_trees_predictions = decision_tree_mod.predict(test_input_attributes)
    decision_tree_score = decision_tree_mod.score(test_input_attributes, test_output_attributes) * 100
else:
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_input_attributes, train_output_attributes)
    # Save Decision Tree's model
    with open("decision_tree_model", "wb") as dt:
        pickle.dump(decision_tree, dt)
    with open("decision_tree_model", "rb") as dt:
        decision_tree_mod = pickle.load(dt)
    decision_trees_predictions = decision_tree.predict(test_input_attributes)
    decision_tree_score = decision_tree.score(test_input_attributes, test_output_attributes) * 100

# Random Forest
if os.path.isfile("./random_forest_model") == True:
    random_forest = RandomForestClassifier(n_estimators= 100)
    with open("random_forest_model", "rb") as rf:
        random_forest_mod = pickle.load(rf)
    random_forest_decisions = random_forest_mod.predict(test_input_attributes)
    random_forest_score = random_forest_mod.score(test_input_attributes, test_output_attributes) * 100
else:
    random_forest = RandomForestClassifier(n_estimators= 100)
    random_forest.fit(train_input_attributes, train_output_attributes)
    # Save Random Forest's model
    with open("random_forest_model", "wb") as rf:
        pickle.dump(random_forest, rf)
    with open("random_forest_model", "rb") as rf:
        random_forest_mod = pickle.load(rf)
    random_forest_decisions = random_forest.predict(test_input_attributes)
    random_forest_score = random_forest.score(test_input_attributes, test_output_attributes) * 100

# Scores of each classifiers and models
print("Random Forest Score: ", random_forest_score)
print("Decision Tree Score: ", decision_tree_score)

# The below function is directly taken from the scikit-learn website to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Plot confusion matrix for Decision Trees
decision_tree_confusion_matrix = confusion_matrix(test_output_attributes, decision_trees_predictions.round())
print("Confusion Matrix - Decision Tree")
print(decision_tree_confusion_matrix)
plot_confusion_matrix(decision_tree_confusion_matrix, classes=[0, 1], title= "Confusion Matrix - Decision Tree")

# Plot confusion matrix for Random Forests
random_forest_confusion_matrix = confusion_matrix(test_output_attributes, random_forest_decisions.round())
print("Confusion Matrix - Random Forest")
print(random_forest_confusion_matrix)
plot_confusion_matrix(random_forest_confusion_matrix, classes=[0, 1], title= "Confusion Matrix - Random Forest")

# The metrics function prints the accuracy, precision, recall,a nd f1-score.
def metrics(actuals, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))
  
print("Evaluation of Decision Tree Model")
metrics(test_output_attributes, decision_trees_predictions.round())
print("Evaluation of Random Forest Model")
metrics(test_output_attributes, random_forest_decisions.round())

# There is a huge imbalance in the dataset between fraudulent and non fraudulent 
# transactions. With such imbalance often comes predictions that favor one
# transaction more than the other, with importance given to genuine transactions.
# In the following lines oversampling method is used to address the imbalance 
# dataset. In this project the minority class is doubled by generating and replicating 
# existing ones. The Synthetic Minority Oversampling Technique (SMOTE) method is data
# augmentation that's used for solving the imbalance problem.
resampled_x, resampled_y = SMOTE().fit_resample(input_attributes, output_attributes)
print("Resampled shape of input_attributes: ", resampled_x.shape)
print("Resampled shape of output_attributes: ", resampled_y.shape)

value_counts = Counter(resampled_y)
print(value_counts)
(train_input_attributes, test_input_attributes, train_output_attributes, test_output_attributes) = train_test_split(resampled_x, resampled_y, test_size= 0.3, random_state= 42)

if os.path.isfile("./resampled_random_forest_model") == True:
    resample_random_forest = RandomForestClassifier(n_estimators = 100)
    with open("resampled_random_forest_model", "rb") as rf:
        resampled_random_forest_mod = pickle.load(rf)
    resampled_predictions = resampled_random_forest_mod.predict(test_input_attributes)
    resampled_random_forest_score = resampled_random_forest_mod.score(test_input_attributes, test_output_attributes) * 100
else:   
    # Build the Random Forest classifier on the new dataset
    resample_random_forest = RandomForestClassifier(n_estimators = 100)
    resample_random_forest.fit(train_input_attributes, train_output_attributes)
    with open("resampled_random_forest_model", "wb") as rf:
        pickle.dump(resample_random_forest, rf)
    with open("resampled_random_forest_model", "rb") as rf:
        resampled_random_forest_mod = pickle.load(rf)
    resampled_predictions = resampled_random_forest_mod.predict(test_input_attributes)
    resampled_random_forest_score = resampled_random_forest_mod.score(test_input_attributes, test_output_attributes) * 100

# Visualize the confusion matrix
#cm_resampled = confusion_matrix(test_output_attributes, y_predict.round())
#print("Confusion Matrix - Random Forest")
#print(cm_resampled)
#plot_confusion_matrix(cm_resampled, classes=[0, 1], title= "Confusion Matrix - Random Forest After Oversampling")

print("Evaluation of Random Forest Model")
print()
metrics(test_output_attributes, resampled_predictions.round())
