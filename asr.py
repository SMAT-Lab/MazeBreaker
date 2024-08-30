import pandas as pd
import numpy as np
import logging
import pandas as pd
from tool import prompt_compose
from agent import MutatePolicy, AgentModel, ModelType
import time
from datetime import datetime

import concurrent.futures
def read_attacked_csv(file_path):
    df = pd.read_csv(file_path)
    text = df['text'].values
    visited = df['visited'].values
    attacked = df['attacked'].values
    score = df['score'].values
    return text, visited, attacked, score

def get_top_10_templates(time, model):
    text, visited, attacked, score = read_attacked_csv(f'./fuzzing/{time}_{model}/template_result.csv')
    score = np.array(score)
    visited = np.array(visited)
    attacked = np.array(attacked)
    text = np.array(text)
    index = np.argsort(visited * 5 + score)[::-1]  
    templates = []

    count = 0
    for i in index:
        print(f'{attacked[i]} / {visited[i]}: {attacked[i] / visited[i]} : {score[i]}')
        templates.append(text[i])
        count += 1
        if count == 15:
            break

    return templates

# [vicuna, deepseek-coder,4o, deepseek-chat, llama3.1, qwen2, claude, glm-4-air, gemma2, llama3]
# attack_time = '20240805_112408'
# target_model_name = "llama2" ('20240829_234126', 'deepseek-coder'), ('20240829_234400', 'glm-4-air'),('20240829_234439', 'gemini-1.5-flash'), ('20240829_234812', 'gemma2'),
attack_time =  '20240829_234439'
target_model_name =  'gemini-1.5-flash'
templates = get_top_10_templates(attack_time, target_model_name)

current_time = attack_time
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(f'./log/MADDPG/ASR_{current_time}_{target_model_name}.log'), 
                              logging.StreamHandler()])

if target_model_name == 'gpt-4o-mini':
    target_model = AgentModel(model_path=target_model_name, model_type=ModelType.openai_llm)
elif target_model_name in ['deepseek-coder', 'deepseek-chat']:
    target_model = AgentModel(model_path=target_model_name, model_type=ModelType.deepseek_llm)
elif target_model_name in [ 'claude-3-5-sonnet-20240620', 'gemini-1.5-flash', 'glm-4-air', 'gpt-3.5-turbo', 'llama-3.1-8b', 'llama-3.1-70b', 'llama-3.1-405b']:
    target_model = AgentModel(model_path=target_model_name, model_type=ModelType.gptgod_llm)
else:
    target_model = AgentModel(model_path=target_model_name, model_type=ModelType.ollama_llm)

judgement_agent = AgentModel(model_path='gpt-4o-mini', model_type=ModelType.openai_llm)
question_seed_init_file = "./experiments/" + 'question_seed.csv'

df = pd.read_csv(question_seed_init_file)
questions = df['text'].values
results = {question: [None] * 15 for question in questions}

def process_question(question, template, template_id):
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            prompt = prompt_compose(question, template)
            res = target_model.run(prompt)
            judge_score = judgement_agent.calculate_reward(res)
            logging.info(f"Question: {question}\nPrompt: {prompt} \nResult: {res} \nJudge Score: {judge_score}")
            results[question][template_id] = judge_score
            return True
        except Exception as e:
            retry_count += 1
            logging.error(f"Error processing question '{question}': {e}")
            if retry_count < MAX_RETRIES:
                logging.info(f"Retrying... ({retry_count}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"Failed to process question '{question}' after {MAX_RETRIES} retries.")
                results[question][template_id] = 0
                return False



for i, template in enumerate(templates):
    ## template's ASR
    logging.info(f'Start to process template {i}')
    MAX_RETRIES = 3
    RETRY_DELAY = 2 

    # for question in questions:
    #     process_question(question, template, i)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_question, question, template, i) for question in questions]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # logging.info(f"Success : {success}, sum: {len(questions)}")

    results_df = pd.DataFrame.from_dict(results, orient='index', columns=[f'Template_{i}' for i in range(15)])
    results_df.index.name = 'Question'
    results_df.reset_index(inplace=True)
    results_df.to_csv(f'attack_results/attack_results_{current_time}.csv', index=False)

logging.info(f"Results written to 'attack_results/attack_results_{current_time}.csv'")