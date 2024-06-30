import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from gensim.utils import simple_preprocess
import csv
from collections import defaultdict
import nltk
import joblib
from transformers import AutoModel
from nltk.tokenize import sent_tokenize
# nltk.download('stopwords')


import os


# -----------------------------------xxxxxxxxxxxxxxx-----------------------------------------------------------------



script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

train_data = pd.read_csv(parent_dir + r"\Testing\train_data_on_queries.csv",header=None)

X = train_data.iloc[:,0:768]

# print(X[0])

y = train_data.iloc[:,768]


def train_log_reg(X_train,y_train):
    classifiers = [LogisticRegression(fit_intercept=False,max_iter=10000,C=0.2,random_state=42) for i in range(7)]
    for i,classifier in enumerate(classifiers):
        result_array = np.where(y_train == i+1, 1, 0)
        classifier.fit(X_train,result_array)
        model_filename = script_dir + '\\model_parameters\\logistic_regression_model' + str(i+1) + '.joblib'

        joblib.dump(classifier, model_filename)


def train_decision_trees(X_train,y_train):
    classifiers = [DecisionTreeClassifier(max_depth=5,min_samples_leaf=2,min_samples_split=5,random_state=42) for i in range(7)]
    for i,classifier in enumerate(classifiers):
        result_array = np.where(y_train == i+1, 1, 0)
        classifier.fit(X_train,result_array)
        model_filename = script_dir + '\\model_parameters\\decision_tree_model' + str(i+1) + '.joblib'

        joblib.dump(classifier, model_filename)

def train_random_forests(X_train,y_train):
    classifiers = [RandomForestClassifier(n_estimators=100, random_state=42,max_depth=5,min_samples_leaf=2,min_samples_split=5,criterion='entropy') for i in range(7)]
    for i,classifier in enumerate(classifiers):
        result_array = np.where(y_train == i+1, 1, 0)
        classifier.fit(X_train,result_array)
        model_filename = script_dir + '\\model_parameters\\random_forest_model' + str(i+1) + '.joblib'

        joblib.dump(classifier, model_filename)



train_log_reg(X,y)
train_decision_trees(X,y)
train_random_forests(X,y)


# classifiers = [DecisionTreeClassifier(max_depth=5,min_samples_leaf=2,min_samples_split=5,random_state=42) for i in range(7)]

# classifiers = [LogisticRegression(fit_intercept=False,max_iter=10000,C=0.2,random_state=42) for i in range(7)]

# classifiers = [RandomForestClassifier(n_estimators=100, random_state=42,max_depth=5,min_samples_leaf=2,min_samples_split=5,criterion='entropy') for i in range(7)]


# for i,classifier in enumerate(classifiers):
#     result_array = np.where(y == i+1, 1, 0)
#     classifier.fit(X,result_array)
#     model_filename = 'random_forest_model' + str(i+1) + '.joblib'

#     joblib.dump(classifier, model_filename)


# classifiers = list()
# for i in range(7):
#     model_filename = 'decision_tree_model' + str(i+1) + '.joblib'
#     classifiers.append(joblib.load(model_filename))






# model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)


# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# test_file = parent_dir + r"\Testing\Manual Queries.txt"
# with open(test_file,'r',encoding='utf-8',newline="") as out_file:
#     user_inputs = out_file.readlines()
#     output_table = [[0 for j in range(7)] for i in range(10)]
#     for i,user_input in enumerate(user_inputs):
#         sentences = list(sent_tokenize(user_input))
#         embeddings = model.encode(sentences)
#         for j,classifier in enumerate(classifiers):
#             outputs = classifier.predict(embeddings)
#             output  = max(outputs)
#             output_table[i][j]=output
#     for i in output_table:
#         print(*i)


# while(True):
#     user_input = input("Please enter your query: ")
#     sentences = list(sent_tokenize(user_input))
#     embeddings = model.encode(sentences)
#     # print(embeddings[1][2])

#     for i,classifier in enumerate(classifiers):
#         outputs = classifier.predict(embeddings)
#         output  = max(outputs)
#         print(f"Prediction of existence in act{i+1} is: ", output)
#     s=input("DO you want to continue?(y/n)")
#     os.system('cls')
#     if(s!='y'):
#         break



# -----------------------------------xxxxxxxxxxxxxxx-----------------------------------------------------------------