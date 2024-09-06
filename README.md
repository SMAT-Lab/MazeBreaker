# FuzzingLLM
This is the official repository for "RatAttacker: Escaping LLM Security Maze for Effective Jailbreaking"
![maze_motivation_with_example.pdf](./figure/maze_motivation_with_example.png)

Here's the framework of our attack approach
![overview.pdf](./figure/overview.png)


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
We use AgileRL to conduct the experiment. The framework of the multi-agent RL is:
![rl.pdf](./figure/rl.png)


You should change the target model name in line 20, and you can reload the attack process by cancel the comment in line 145.
```shell
python RL_MADDPG.py
```

We provide the average accumulated_reward of the successful attack when attacking Deepseek-chat as follow:
![accumulated_reward.pdf](./figure/accumulated_reward.png)
## calculate the ASR
Set the attack time and the target model name in asr.py, and run the following code to calculate the ASR.
```shell
python asr.py
```

## calculate the attack data
We give the script in script folder to calculate the attack data. 
