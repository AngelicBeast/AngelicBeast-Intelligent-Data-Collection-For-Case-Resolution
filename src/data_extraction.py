import os
from transformers import AutoModel
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
import csv





#Dataset Directory Path:
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
dataset_path = parent_dir + r"\Dataset\Indian_Acts"

#Output Path:
output_path = parent_dir + r"\Extracted Data\data.csv"

#Clearing the existing Data
with open(output_path,'r',encoding='utf-8') as obj:
    pass



#Loading the jina Embedding Model:
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)



#writing data into csv file:

def write_data(embeddings,label):
    with open(output_path,'a',encoding='utf-8',newline="") as out_file:
        csv_writer = csv.writer(out_file)
        for i in embeddings:
            i = list(i)
            i.append(label)
            csv_writer.writerow(i)




#Function to read all the text files in the directory:
def read_text_files(directory_path):
    # Get the list of files in the directory
    file_list = os.listdir(directory_path)

    # Filter out non-text files if necessary
    text_files = [file for file in file_list if file.endswith(".txt")]

    label=7
    # Read the content of each text file
    for text_file in text_files:
        file_path = os.path.join(directory_path, text_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            sentences = sent_tokenize(content)
            embeddings = model.encode(sentences)
            write_data(embeddings,label)

        label+=1
        print(label-1," Done")


read_text_files(dataset_path)





