# FuzzingLLM

## Environment setup

GPU server with CUDA 12.2 is required.

```shell
conda create --name Fuzzing python=3.9 -y
conda activate Fuzzing
pip install -r requirements.txt
```

## LLM's interface
We use [GPTGod](https://gptgod.online/), [deepseek](https://platform.deepseekc.om), [ollama](https://www.ollama.com), [openai](https://platform.openai.com)  as the interface of LLM. You can use the following code to interact with LLM.

You should input the api-key in tool.py
You should input the api-key in jailbreak_evaluation.py in line 38
and set the api key in llm/llm.py

## single RL
```shell
python RL.py
```
## MARL
We use AgileRL to conduct the experiment. You should change the target model name in line 20, and you can reload the attack process by cancel the comment in line 145.
```shell
python RL_MADDPG.py
```
## calculate the ASR
Set the attack time and the target model name in asr.py, and run the following code to calculate the ASR.
```shell
python asr.py
```

## calculate the attack data
We give the script in script folder to calculate the attack data. 