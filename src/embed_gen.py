import os
import csv
from transformers import AutoModel


#Loading the model
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
# -----------------------------xxxxx-----------------------------------------------------------


from nltk.tokenize import sent_tokenize

def remove_empty_lines(lst):
    indices=[]
    for i in range(len(lst)):
        if(lst[i]!=""):
            indices.append(i)
    lst_updated = []
    for j in indices:
        lst_updated.append(lst[j])
    return lst_updated

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
out_file_test = os.path.dirname(curr_dir) + r"\Testing\test_data.csv"
out_file_train = os.path.dirname(curr_dir) + r"\Testing\train_data_on_queries.csv"

with open(out_file_test,'w',encoding='utf-8',newline="") as out_file_obj:
    pass

with open(out_file_train,'w',encoding='utf-8',newline="") as out_file_obj:
    pass

for counter,file in enumerate(os.listdir(parent_dir + r"\Testing")):
    if file.endswith(".txt") and file.startswith("test"):
        _dot = file.find('.')
        label = int(file[13:_dot])
        with open(parent_dir + r"\Testing\\" + file,'r',encoding='utf-8') as file_obj:
            queries_= [i.split('. ',1) for i in file_obj.readlines()]
            queries = [i[-1].rstrip("\n") for i in queries_]
            queries = remove_empty_lines(queries)
            embeddings = model.encode(queries)
            with open(out_file_test,'a',encoding='utf-8',newline="") as out_file_obj:
                csv_writer = csv.writer(out_file_obj)
                for embedding in embeddings:
                    embedding = list(embedding)
                    embedding.append(label)
                    csv_writer.writerow(embedding)
        print(f"{file} done")


for counter,file in enumerate(os.listdir(parent_dir + r"\Testing\\")):
    if file.endswith(".txt") and file.startswith("train"):
        _dot = file.find('.')
        label = int(file[14:_dot])
        with open(parent_dir + r"\Testing\\" + file,'r',encoding='utf-8') as file_obj:
            queries_= [i.split('. ',1) for i in file_obj.readlines()]
            queries = [i[-1].rstrip("\n") for i in queries_]
            queries = remove_empty_lines(queries)
            embeddings = model.encode(queries)
            with open(out_file_train,'a',encoding='utf-8',newline="") as out_file_obj:
                csv_writer = csv.writer(out_file_obj)
                for embedding in embeddings:
                    embedding = list(embedding)
                    embedding.append(label)
                    csv_writer.writerow(embedding)
        print(f"{file} done")
        
        


                          
                    

            



