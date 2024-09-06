import numpy as np

import os

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation
import torch

from Fuzzing_v1 import Fuzzing_v1

def load_checkpoint(agent, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        agent.loadCheckpoint(filename)
    else:
        print(f"No checkpoint found at {filename}")

target_model_name = 'qwen2' # "claude-3-5-sonnet-20240620" # "llama-3.1-8b"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL MADDPG Demo =====")

    # Define the network configuration
    NET_CONFIG = {
        "arch": "mlp",
        "hidden_size": [128, 128]
    }

    INIT_HP = {
        "POPULATION_SIZE": 2,
        "ALGO": "MADDPG",
        "CHANNELS_LAST": True,
        "BATCH_SIZE": 8,
        "LR_ACTOR": 0.001,
        "LR_CRITIC": 0.01,
        "GAMMA": 0.95,
        "MEMORY_SIZE": 10000,
        "LEARN_STEP": 5,
        "TAU": 0.01,
    }

    env = Fuzzing_v1.parallel_env()
    # if INIT_HP["CHANNELS_LAST"]:
    #     env = ss.frame_skip_v0(env, 4)
    #     env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    #     env = ss.color_reduction_v0(env, mode="B")
    #     env = ss.resize_v1(env, x_size=84, y_size=84)
    #     env = ss.frame_stack_v1(env, 4)
    env.reset()

    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[0], 1, 1) for state_dim in state_dim
        ]

    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    pop = initialPopulation(
        INIT_HP["ALGO"],
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        device=device,
    )

    # Load saved checkpoints if available
    # load_step = 1700

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=INIT_HP["POPULATION_SIZE"],
        evo_step=1,
    )

    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=0.2,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0,
        rl_hp=0.2,
        rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
        ],
        mutation_sd=0.1,
        min_lr=0.0001,
        max_lr=0.01,
        min_learn_step=1,
        max_learn_step=120,
        min_batch_size=8,
        max_batch_size=64,
        agent_ids=INIT_HP["AGENT_IDS"],
        arch=NET_CONFIG["arch"],
        rand_seed=1,
        device=device,
    )

    max_episodes = 50
    max_steps = 100
    epsilon = 1.0
    eps_end = 0.1
    eps_decay = 0.995
    evo_epochs = 20
    evo_loop = 1
    elite = pop[0]

    
    # checkpoint_path = f"./models/{target_model_name}/MADDPG/MADDPG_trained_agent_8.pt"
    # load_checkpoint(elite, checkpoint_path)
    # target_model_name = "llama-3.1-405b"

    for idx_epi in range(max_episodes):
        for agent in pop:
            state, info = env.reset()
            agent_reward = {agent_id: 0 for agent_id in env.agents}
            # if INIT_HP["CHANNELS_LAST"]:
            #     state = {
            #         agent_id: np.moveaxis(np.expand_dims(s, 0), [-1], [-3])
            #         for agent_id, s in state.items()
            #     }
            for step in range(max_steps):
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )

                cont_actions, discrete_action = agent.getAction(
                    state, epsilon, agent_mask, env_defined_actions
                )
                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                next_state, reward, termination, truncation, info = env.step(action)

                if INIT_HP["CHANNELS_LAST"]:
                    state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}

                memory.save2memory(state, cont_actions, reward, next_state, termination)

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                if (memory.counter % agent.learn_step == 0) and (
                    len(memory) >= agent.batch_size
                ):
                    experiences = memory.sample(agent.batch_size)
                    agent.learn(experiences)

                if INIT_HP["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.expand_dims(ns, 0)
                        for agent_id, ns in next_state.items()
                    }
                state = next_state

                if any(truncation.values()) or any(termination.values()):
                    break

            score = sum(agent_reward.values())
            agent.scores.append(score)

        path = f"./models/{target_model_name}/MADDPG"
        filename = f"MADDPG_trained_agent_{idx_epi}.pt"
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        elite.saveCheckpoint(save_path)

        epsilon = max(eps_end, epsilon * eps_decay)

        if (idx_epi + 1) % evo_epochs == 0:
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {idx_epi + 1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f" % fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f" % np.mean(agent.fitness[-100:]) for agent in pop]}'
            )

            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

    path = f"./models/{target_model_name}/MADDPG"
    filename = "MADDPG_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.saveCheckpoint(save_path)
