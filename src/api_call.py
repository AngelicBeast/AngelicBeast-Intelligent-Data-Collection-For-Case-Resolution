from openai import OpenAI
import openai
import os
import csv



#Function that returns a list of text files in a directory
def list_txt_files(directory):
    txt_files = []
    for file in os.listdir(directory):
        # print(file)
        if file.endswith(".txt"):
            txt_files.append(file)
    return txt_files



#Loading chatGPT API

client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)
def get_completion(prompt="", model="gpt-3.5-turbo",mess=list()):
    messages=list()
    if(len(mess)==0):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = mess
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content



# ---------------------------------------xxxxxx-----------------------------------------



curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)


# Loading act numbers
actno_path = parent_dir + r"\Extracted Data\act_no.csv"
act_no = dict()

with open(actno_path,'r',encoding='utf-8') as actno_obj:
    csv_reader = csv.reader(actno_obj)
    for row in csv_reader:
        act_no[row[0]+".txt"] = int(row[1])


# ---------------------------------------xxxxxx-----------------------------------------


dataset_path = parent_dir + r"\Extracted Data\\"
acts_dataset_path = parent_dir + r"\Dataset\Indian_Acts"
acts_list=list_txt_files(acts_dataset_path)
test_data_list = ["test_data_" + str(i+1) + ".txt" for i in range(7)]
train_data_list = ["train_data_" + str(i+1) + ".txt" for i in range(7)]


for j,act_name in enumerate(acts_list):
    test_data_path = dataset_path  + test_data_list[act_no[act_name]-1]
    out_path = parent_dir + r"\Testing\test_queries_"+  str(act_no[act_name]) + r".txt"
    with open(out_path,'w',encoding='utf-8') as obj:
        pass
    act=""
    with open(test_data_path,'r',encoding="utf-8") as obj:
        act = obj.read()

    sz = len(act)
    n = sz//(3000)
    chunks = list()

    for i in range(n+1):
        chunks.append(act[3000*i:min(3000*(i+1),sz)])
    

    for chunk in chunks:
        mess = [{'role':'system','content':"You are an intelligent user query generator. You will be given an Indian Government Law. You have to read it properly, and then generate 10 queries by becoming a user who wants to file a case in the court. The queries should be real life problems, related to the Law. Also you have to ouput queries such that the output given by you can be directly converted to a text file."},{'role':'user','content':f'Act:\n{chunk}'},{'role':'system','content':"Note that: the use of the words: \"the Act\" and \"this law\" in the queries is strictly prohibited. Also, the queries sound like the person wants to file a court case, they should not be general queries regarding the law."}]

        text = get_completion(mess=mess)
        
        with open(out_path,'a',encoding='utf-8') as write_obj:
            write_obj.write(text+"\n")
    print(f"{act_name} done")


    

for j,act_name in enumerate(acts_list):
    train_data_path = dataset_path  + train_data_list[act_no[act_name]-1]
    out_path = parent_dir + r"\Testing\train_queries_"+  str(act_no[act_name]) + r".txt"
    with open(out_path,'w',encoding='utf-8') as obj:
        pass
    act=""
    with open(train_data_path,'r',encoding="utf-8") as obj:
        act = obj.read()

    sz = len(act)
    n = sz//(3000)
    chunks = list()

    for i in range(n+1):
        chunks.append(act[3000*i:min(3000*(i+1),sz)])

    for chunk in chunks:
        mess = [{'role':'system','content':"You are an intelligent user query generator. You will be given an Indian Government Law. You have to read it properly, and then generate 10 queries by becoming a user who wants to file a case in the court. The queries should be real life problems, related to the Law. Also you have to ouput queries such that the output given by you can be directly converted to a text file."},{'role':'user','content':f'Act:\n{chunk}'},{'role':'system','content':"Note that: the use of the words: \"the Act\" and \"this law\" in the queries is strictly prohibited. Also, the queries sound like the person wants to file a court case, they should not be general queries regarding the law."}]
        text = get_completion(mess=mess)
        
        with open(out_path,'a',encoding='utf-8') as write_obj:
            write_obj.write(text+"\n")
    print(f"{act_name} done")


    