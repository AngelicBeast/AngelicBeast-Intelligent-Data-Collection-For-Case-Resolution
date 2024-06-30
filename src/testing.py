import joblib
import os
import csv
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


classifiers_dec = list()
classifiers_log = list()
classifiers_rf = list()
for i in range(7):
    model_filename_dec = 'decision_tree_model' + str(i+1) + '.joblib'
    model_filename_log = 'logistic_regression_model' + str(i+1) + '.joblib'
    model_filename_rf = 'random_forest_model' + str(i+1) + '.joblib'
    classifiers_dec.append(joblib.load(model_filename_dec))
    classifiers_log.append(joblib.load(model_filename_log))
    classifiers_rf.append(joblib.load(model_filename_rf))


#file paths:
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
test_file = os.path.dirname(curr_dir) + r"\Testing\test_data.csv"


# Loading act numbers
actno_path = parent_dir + r"\Extracted Data\act_no.csv"

act_no_inv = dict()

with open(actno_path,'r',encoding='utf-8') as actno_obj:
    csv_reader = csv.reader(actno_obj)
    for row in csv_reader:
        act_no_inv[int(row[1])] = row[0]
# -----------------------xxxx------------------------------------------

test_data = pd.read_csv(test_file,header=None)

X = test_data.iloc[:,0:768]

# print(X[0][0]+1)
y = test_data.iloc[:,768]





# embedding=""
# with open(test_file,'r',encoding='utf-8') as obj:
#     csv_reader = csv.reader(obj,delimiter=',')
#     for row in csv_reader:
#         embedding = row
#         break
# embedding.pop()

# for i,classifier in enumerate(classifiers_dec):
#     print(f"{i+1}: {classifier.predict([embedding])}")




predictions_dec,predictions_log,predictions_rf = list(),list(),list()

print("\n\n\n")

for i,classifier in enumerate(classifiers_dec):
    true_labels = np.where(y == i+1, 1, 0)
    predicted_labels = classifier.predict(X)
    predictions_dec.append(predicted_labels)
    accuracy = accuracy_score(true_labels,predicted_labels)
    print(f"Accuracy of decision trees for {act_no_inv[i+1]} is: {accuracy}")

print("\n\n\n")

for i,classifier in enumerate(classifiers_log):
    true_labels = np.where(y == i+1, 1, 0)
    predicted_labels = classifier.predict(X)
    predictions_log.append(predicted_labels)
    accuracy = accuracy_score(true_labels,predicted_labels)
    print(f"Accuracy of logistic regression for {act_no_inv[i+1]} is: {accuracy}")


print("\n\n\n")
for i,classifier in enumerate(classifiers_rf):
    true_labels = np.where(y == i+1, 1, 0)
    predicted_labels = classifier.predict(X)
    predictions_rf.append(predicted_labels)
    accuracy = accuracy_score(true_labels,predicted_labels)
    print(f"Accuracy of random forest for {act_no_inv[i+1]} is: {accuracy}")

print("\n\n\n")
