import pandas as pd
import numpy as np
import logging
import pandas as pd
from tool import prompt_compose
from agent import MutatePolicy, AgentModel, ModelType
import time
from datetime import datetime

import concurrent.futures

def read_template_csv(file_path):
    df = pd.read_csv(file_path)
    templates = df['Template'].values
    return templates

templates = read_template_csv('script/transfer_attack/top_templates_asr.csv')

models = ['gemini-1.5-flash']# ['gpt-4o', 'gpt-4o-mini', 'deepseek-coder', 'deepseek-chat', 'claude-3-5-sonnet-20240620', 'glm-4-air', 'gpt-3.5-turbo', 'llama-3.1-8b', 'llama-3-70b', 'llama-3.1-70b']

for target_model_name in models:
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f'./log/transfer_attack_{current_time}_{target_model_name}.log'), 
                                  logging.StreamHandler()])

    if target_model_name in ['gpt-4o-mini', 'gpt-4o']:
        target_model = AgentModel(model_path=target_model_name, model_type=ModelType.openai_llm)
    elif target_model_name in ['deepseek-coder', 'deepseek-chat']:
        target_model = AgentModel(model_path=target_model_name, model_type=ModelType.deepseek_llm)
    elif target_model_name in [ 'claude-3-5-sonnet-20240620', 'gemini-1.5-flash', 'glm-4-air', 'gpt-3.5-turbo', 'llama-3.1-8b', 'llama-3-70b', 'llama-3.1-70b', 'llama-3.1-405b', 'mixtral-8x22b', 'qwen-72b']:
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
        # print(f'{i}: {template}')
        logging.info(f'Start to process template {i}')
        # success = 0
        MAX_RETRIES = 3
        RETRY_DELAY = 2 

        # for question in questions:
        #     process_question(question, template, i)

        max_threads = 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(process_question, question, template, i) for question in questions]
            for future in concurrent.futures.as_completed(futures):
                future.result()


        # logging.info(f"Success : {success}, sum: {len(questions)}")

        results_df = pd.DataFrame.from_dict(results, orient='index', columns=[f'Template_{i}' for i in range(15)])
        results_df.index.name = 'Question'
        results_df.reset_index(inplace=True)
        results_df.to_csv(f'./srcipt/transfer_attack/attack_results_{current_time}_{target_model_name}.csv', index=False)

    logging.info(f"Results written to 'attack_results_{current_time}.csv'")