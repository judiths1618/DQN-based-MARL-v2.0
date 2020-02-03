import matplotlib.pyplot as plt
import numpy as np

from DQN import DeepQNetwork
from env import Env
from memory import Memory
from sklearn.preprocessing import StandardScaler
import pandas as pd
import chart_studio.plotly as py
import plotly.figure_factory as ff
import datetime, random


N_VM = 10
EPISODES = 500
MINI_BATCH = 128
MEMORY_SIZE = 10000
REPLACE_TARGET_ITER = 100
N_AGENT = 2


def run_env():
    step = 0
    for episode in range(EPISODES):
        rwd = [0.0, 0.0]
        obs = env.reset()
        # print(episode)

        while True:
            step += 1
            q_value = []
            if np.random.uniform() < dqn[0].epsilon:
                for i in range(N_AGENT):
                    # TODO(hang): standardized q_value
                    # normalize = scaler.fit_transform(dqn[i].choose_action(obs[i]).reshape(-1, 1)).reshape(7)
                    # q_value.append(normalize)
                    q_value.append(dqn[i].choose_action(obs[i]))
                
                # TODO: 如何处理两个agent的q值：joint q
                q_joint = []
                for i in range(len(q_value[0])):
                    # communicate_strategy =
                    q_joint.append((q_value[0][i] + q_value[1][i]))
                action = np.argmax(q_joint)
                # actions = [np.argmax(q_value[0]), np.argmax(q_value[1])]
                # fitness = [env.compute(actions[0]), env.compute(actions[1])]
                # if random.random() < 0.7:
                #     action = actions[0]
                # else:
                #     action = actions[1]
                # print(np.argmax(q_value[0]), np.argmax(q_value[1]), action)
            else:
                action = np.random.randint(0, env.n_actions - 1)

            obs_, reward, done = env.step(action)
            # print(obs_)
            rwd[0] += reward[0]
            rwd[1] += reward[1]

            # for i in range(N_AGENT):
            #     memories[i].remember(obs, [action, action], reward[i], obs_, done[i])
            #     size = memories[i].pointer
            #     batch = random.sample(range(size), size) if size < MINI_BATCH else random.sample(
            #         range(size), MINI_BATCH)
            #     if step > 200 and step % 5 == 0:
            #         dqn[i].learn(*memories[i].sample(batch, N_AGENT))

            for i in range(N_AGENT):
                memories[i].remember(obs[i], action, reward[i], obs_[i], done[i])
                size = memories[i].pointer
                batch = random.sample(range(size), size) if size < MINI_BATCH else random.sample(
                    range(size), MINI_BATCH)
                if step > REPLACE_TARGET_ITER and step % 5 == 0:
                    dqn[i].learn(*memories[i].sample(batch))

            obs = obs_


            if done[0]:
                
                
                if episode % 10 == 0:
                    print(
                            'episode:' + str(episode) + ' steps:' + str(step) +
                            ' reward0:' + str(round(rwd[0],6)) + ' reward1:' + str(round(rwd[1],6)) +
                            ' eps_greedy0:' + str(round(dqn[0].epsilon,6)) + ' eps_greedy1:' + str(round(dqn[1].epsilon,6)), 
                            bool(env.strategies)
                        )
                
                if episode == EPISODES - 1:
                    print(episode, bool(env.strategies))
                    print(max(env.vm_time))
                    print(np.sum(env.vm_cost))
                    print(env.vm_time)
                    print(env.vm_cost/3600)

                    # env.strategies

                for i in range(N_AGENT):
                    rewards[i].append(rwd[i])
                # rewards[0].append(max(env.vm_time))
                # rewards[1].append(np.sum(env.vm_cost))
                break


if __name__ == '__main__':
    rewards = [[], []]

    scaler = StandardScaler()

    env = Env(N_VM, N_AGENT)

    memories = [Memory(MEMORY_SIZE) for i in range(N_AGENT)]
    memory = Memory(MEMORY_SIZE)

    dqn = [DeepQNetwork(env.n_actions,
                        env.n_features,
                        i,
                        learning_rate=0.0001,
                        replace_target_iter=REPLACE_TARGET_ITER,
                        e_greedy_increment=2e-5
                        ) for i in range(N_AGENT)]

    run_env()
    
    print()
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.grid(True)
    ax0.set_xlabel('episodes')
    ax0.set_ylabel('Reward 1(makespan metric)')
    ax0.plot([i for i in range(len(rewards[0]))], rewards[0], label='makespan')
    ax0.legend()
    ax1.grid(True)
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('Reward 2(cost metric)')
    ax1.plot([i for i in range(len(rewards[1]))], rewards[1], 'orange', label='cost')
    ax1.legend()
    fig.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    # plt.savefig ("C:\\Users\\Judiths\\Desktop\\Access图表\\Fig4\\" + 'makespan&cost_conv_dqn.svg')
    plt.show()