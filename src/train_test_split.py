import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import csv


no_of_acts = 7



data = pd.read_csv(r"C:\Users\isart\OneDrive\Desktop\CSC Project\Extracted Data\data.csv",header=None)

X = data.iloc[:,0:768]

# print(X[0])

y = data.iloc[:,768]



X_chunked = []
y_chunked=[]

ii=0
ind=1
label=1
temp_X=list()
temp_y=list()
while(ii<len(X)):
    temp_X.append(list(X.iloc[ii,:].values))
    temp_y.append((y.iloc[ii]))
    if(label!=temp_y[-1]):
        label = temp_y[-1]
        x_temp,y_temp = temp_X.pop(),temp_y.pop()
        X_chunked.append([ind,]+temp_X)
        y_chunked.append([ind,]+temp_y)
        temp_y = [y_temp]
        temp_X = [x_temp]
        ind=1
    elif(ii%5==4 ):
        X_chunked.append([ind,]+temp_X)
        y_chunked.append([ind,]+temp_y)
        temp_y = []
        temp_X = []
        ind+=1
    ii+=1

X_chunked.append([ind,]+temp_X)
y_chunked.append([ind,]+temp_y)


X_train_chunked, X_test_chunked, y_train_chunked, y_test_chunked = train_test_split(X_chunked, y_chunked, test_size=0.2, random_state=42)



X_train=[]
y_train= []

for i in range(len(X_train_chunked)):
    for j in range(1,len(X_train_chunked[i])):
        X_train.append(X_train_chunked[i][j])
        y_train.append(y_train_chunked[i][j])





X_train = np.array(X_train)
y_train = np.array(y_train)






script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
extracted_data_path = parent_dir + r"\Extracted Data"
train_data_embedddings_file = extracted_data_path + r"\train_data.csv"

test_data_files = [extracted_data_path + r"\test_data_" + str(i+1) + r".txt" for i in range(no_of_acts)]
train_data_files = [extracted_data_path + r"\train_data_" + str(i+1) + r".txt" for i in range(no_of_acts)]




def file_writer(file_path,X,y):
    with open(file_path,'w',encoding='utf-8',newline="") as obj:
        csv_writer = csv.writer(obj)
        for ind,i in enumerate(X):
            i = list(i)
            i.append(y[ind])
            csv_writer.writerow(i)

file_writer(train_data_embedddings_file,X_train,y_train)

entire_data=[[] for i in range(no_of_acts)]



curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)


# Loading act numbers
actno_path = parent_dir + r"\Extracted Data\act_no.csv"

act_no = dict()

with open(actno_path,'r',encoding='utf-8') as actno_obj:
    csv_reader = csv.reader(actno_obj)
    for row in csv_reader:
        act_no[row[0] + r".txt"] = int(row[1])



file_list = os.listdir(parent_dir + r"\Dataset\Indian_Acts")

text_files = [file for file in file_list if file.endswith(".txt")]

for file in text_files:
    with open(parent_dir + r"\Dataset\Indian_Acts\\"+file,"r",encoding="utf-8") as obj:
        entire_text = list(obj.readlines())
        entire_text = [entire_text[i] for i in range(len(entire_text))]
        entire_data[act_no[file]-1] = entire_text



for i,chunk in enumerate(X_test_chunked):
    ind = chunk[0]
    label = y_test_chunked[i][1]
    lines = entire_data[label-1][ind*5:ind*5+len(chunk)]
    with open(test_data_files[label-1],"a",encoding="utf-8") as obj:
        obj.writelines(lines+["\n",])

for i,chunk in enumerate(X_train_chunked):
    ind = chunk[0]
    label = y_train_chunked[i][1]
    lines = entire_data[label-1][ind*5:ind*5+len(chunk)]
    with open(train_data_files[label-1],"a",encoding="utf-8") as obj:
        obj.writelines(lines+["\n",])




