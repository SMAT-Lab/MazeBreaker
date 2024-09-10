from transformers import RobertaForSequenceClassification, RobertaTokenizer
import os
import torch
import requests

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("zhx123/ftrobertallm", cache_dir='/home/models', trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("zhx123/ftrobertallm", cache_dir='/home/models', trust_remote_code=True)
# model_path = 'hubert233/GPTFuzz'
# model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir='/home/models', trust_remote_code=True)
# tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir='/home/models', trust_remote_code=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def judge(text):
    # move the model to GPU
    model.to(device)
    
    # Tokenize the input data
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # move the input to GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Make sure the model is in evaluation mode
    model.eval()
    
    # Predict the labels
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert logits to predicted labels
    predicted_labels = torch.argmax(logits, dim=1)
    
    # Convert tensor to list of labels
    predicted_labels = predicted_labels.tolist()

    # print(predicted_labels[0])
    
    # Output the predicted labels
    return predicted_labels[0]


import re
import pandas as pd

question_seed_init_file = '../../experiments/question_seed.csv'
def extract_results_and_scores(log_file_path):
    # read the initial question seed file
    df_seed = pd.read_csv(question_seed_init_file)
    questions = df_seed['text'].values
    
    with open(log_file_path, 'r') as file:
        log_data = file.read()
    
    template_pattern = r"Start to process template \d+"
    question_pattern = r"INFO\s+-\s+Question:\s(.*?)\n.*?Result:\s*(.*?)\s*Judge Score:"
    
    # get every template's results
    templates = re.split(template_pattern, log_data)[1:]
    
    df = pd.DataFrame(index=[f'Template {i+1}' for i in range(len(templates))],
                      columns=questions)
    
    for i, template_content in enumerate(templates):
        results = re.findall(question_pattern, template_content, re.DOTALL)
        
        result_dict = {}
        for question, result in results:
            result_dict[question.strip()] = result.strip()
        
        for question in questions:
            df.at[f'Template {i+1}', question] = result_dict.get(question, 'N/A')
        
        if len(results) != 100:
            print(f"Warning: Template {i + 1} does not have 100 questions, found {len(results)} questions instead.")
    
    return df


import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file(file):
    if not file.endswith('.log'):
        return f"Skipped {file}: Not a log file"

    if not file.startswith('ASR'):
        return f"Skipped {file}: Not an ASR log"

    print(f"Processing {file}...")
    log_file_path = f'../../log/MADDPG/{file}'
    model_name = file.split('_')[-1].split('.')[0]

    try:
        df = extract_results_and_scores(log_file_path)
    except Exception as e:
        return f"Failed to process {file}: {e}"

    result_df = pd.DataFrame(index=df.index, columns=df.columns)
    SAVE_FILE = f'./results/{model_name}_result_df.csv'

    for i in df.index:
        for j in df.columns:
            text = df.at[i, j] 
            try:
                result_df.at[i, j] = judge(text)  # judge the response
            except Exception as e:
                print(e)

            result_df.to_csv(SAVE_FILE, index=True)

    result_df.to_csv(SAVE_FILE, index=True)
    return f"Successfully processed {file}"

files = sorted(os.listdir('../../log/MADDPG/'))

with ThreadPoolExecutor(max_workers=11) as executor:
    future_to_file = {executor.submit(process_file, file): file for file in files}

    for future in as_completed(future_to_file):
        file = future_to_file[future]
        try:
            result = future.result()
            print(result)
        except Exception as exc:
            print(f"{file} generated an exception: {exc}")
