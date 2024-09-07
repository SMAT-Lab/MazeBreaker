import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
import google.generativeai as palm
import httpx
import requests
from tool import OpenAI_API, OpenAI_BASE_URL
# from vllm import LLM as vllm
# from vllm import SamplingParams

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")
    
class GPTGodLLM(LLM):
    def __init__(self,
                 model_path='gpt-3.5-turbo',
                 api_key=None,
                 system_message=None
                ):
        self.api_key = api_key
        self.model = model_path
        self.system_message = system_message

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.gptgod.online/v1",
        )
    

    def generate(self, prompt, temperature=0.7):
        system_message = self.system_message if self.system_message is not None else "You are a helpful assistant."
        messages =  [{"role": "system", "content": system_message}, 
                         {"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )

        return completion.choices[0].message.content

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class OllamaLLM(LLM):
    def __init__(self,
                 model_path='llama3:70b ',
                 api_key='sk-123',
                 system_message=None
                ):
        self.api_key = api_key
        self.model = model_path
        self.system_message = system_message

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        # self.client = OpenAI(api_key=api_key)

    def generate(self, prompt, temperature=0.7):
        client = OpenAI(
            # base_url='http://localhost:11434/v1/',
            base_url='http://localhost:11434/v1/',
            api_key='ollama'
        )

        system_message = self.system_message if self.system_message is not None else "You are a helpful assistant."
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, 
                         {"role": "user", "content": prompt}],
            model=self.model,
            temperature=temperature,
            timeout=40
        )

        return chat_completion.choices[0].message.content

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class DeepSeekLLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()
        self.client = OpenAI(
            base_url='https://api.deepseek.com/v1/',
            api_key='',
        )

        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=1024, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                if n == 1:
                    return [results.choices[i].message.content for i in range(n)][0]
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()
        self.client = OpenAI(
            base_url=OpenAI_BASE_URL,
            api_key=OpenAI_API,
        )

        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=1024, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                if n == 1:
                    return [results.choices[i].message.content for i in range(n)][0]
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
