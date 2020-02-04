import matplotlib.pyplot as plt
import numpy as np

from DQN import DeepQNetwork
from env import Env
from memory import Memory
from workflows.XMLProcess import XMLtoDAG
from sklearn.preprocessing import StandardScaler
import pandas as pd
import chart_studio.plotly as py
import plotly.figure_factory as ff
import datetime, random


N_VM = 10
EPISODES = 500
MINI_BATCH = 128
MEMORY_SIZE = 10000
REPLACE_TARGET_ITER = 200
N_AGENT = 2


def run_env():       # 算法的控制流程
    step = 0
    
    # TODO：添加训练的反馈机制 feedback; 选出对策略其决定性作用的因素，调整它（们）以得到更优的策略; 代代控制？每代控制？
    history_quads = []

    for episode in range(EPISODES):
        rwd = [0.0, 0.0]
        obs = env.reset()
        # print(episode)
        
        while True:
            step += 1
            q_value = []
            # 随机探索机制开始了
            if np.random.uniform() < dqn[0].epsilon:
                for i in range(N_AGENT):
                    q_value.append(dqn[i].choose_action(obs[i]))  
                # TODO: 如何处理两个agent的q值：joint q
                q_joint = []
                for i in range(len(q_value[0])):
                    q_joint.append(np.sqrt((q_value[0][i] * q_value[1][i])))
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
            # print('obs', obs_)
            
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
                batch = random.sample(range(size), size) if size < MINI_BATCH else random.sample(range(size), MINI_BATCH)
                if step > REPLACE_TARGET_ITER and step % 5 == 0:
                    dqn[i].learn(*memories[i].sample(batch))

            obs = obs_

            # 训练结果调度完毕，开始每轮的feedback 
            if done[0]:     
                if episode % 10 == 0:
                    print(
                            'episode:' + str(episode) + ' steps:' + str(step) +
                            ' reward0:' + str(round(rwd[0],6)) + ' reward1:' + str(round(rwd[1],6)) +
                            ' eps_greedy0:' + str(round(dqn[0].epsilon,6)) + ' eps_greedy1:' + str(round(dqn[1].epsilon,6)), 
                            max(env.vm_time),   # makespan
                            np.sum(env.vm_cost) # total cost
                        )
                
                if episode == EPISODES - 1:     # 到了最后一代          
                    print(episode, bool(env.strategies))
                    print(max(env.vm_time))
                    print(np.sum(env.vm_cost))
                    print(env.vm_time)
                    print(env.vm_cost/3600)
                    print(env.strategies)

                for i in range(N_AGENT):
                    rewards[i].append(rwd[i])       
                records[0].append(max(env.vm_time)/100)     # makespan的记录值
                records[1].append(np.sum(env.vm_cost)*10)   # cost的记录值
                break


if __name__ == '__main__':
    rewards = [[], []]          # makespan agent 和 cost agent的奖励函数
    records = [[], []]          # makespan agent 和 cost agent的数值函数
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
    
    print("the trends of rewards between makespan agent and cost agent")
    eps = [i for i in range(len(rewards[0]))]
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.grid(True)
    ax0.set_xlabel('episodes')
    ax0.set_ylabel('Reward 1(makespan metric)')
    ax0.plot(eps, rewards[0], eps, records[0], label='makespan')
    # ax0.plot([i for i in range(len(records[0]))], records[0], label='makespan')
    ax0.legend()
    ax1.grid(True)
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('Reward 2(cost metric)')
    ax1.plot(eps, rewards[1], eps, records[1], 'orange', label='cost')
    ax1.legend()
    fig.subplots_adjust(hspace=0.3)
    fig.tight_layout()
    # plt.savefig ("C:\\Users\\Judiths\\Desktop\\Access图表\\Fig4\\" + 'makespan&cost_conv_dqn.svg')
    plt.show()

    print(env.strategies)
    print('DQN Gantt图========')

    c_tmp = pd.read_excel(".//data//WSP_dataset.xlsx", sheet_name="Containers Price")
    CONTAINERS = list(c_tmp.loc[:, 'Configuration Types'])
    vms = CONTAINERS

    num_mc = len(vms)
    N = [XMLtoDAG('.//workflows//Sipht_29.xml', 29).jobs(), XMLtoDAG('.//workflows//Montage_25.xml', 25).jobs(), XMLtoDAG('.//workflows//Inspiral_30.xml', 30).jobs(), XMLtoDAG('.//workflows//Epigenomics_24.xml', 24).jobs(), XMLtoDAG('.//workflows//CyberShake_30.xml', 30).jobs()]

    time_tmp = pd.read_excel(".//data//WSP_dataset.xlsx", sheet_name="Containers Performance", index_col=[0])
    cost_tmp = pd.read_excel(".//data//WSP_dataset.xlsx", sheet_name="Containers Performance", index_col=[0])
    timematrix = [list(map(float, time_tmp.iloc[i])) for i in range(num_mc)]
    costmatrix = [list(map(float, cost_tmp.iloc[i])) for i in range(num_mc)]
                
    time_mcp = timematrix
    cost_mcp = costmatrix         

    num_job = 5  # number of jobs
    num_mc = 10  # number of machines
    m_keys = [j + 1 for j in range (num_mc)]
    j_keys = [j for j in range (num_job)]
    df = []

    record = []
    for k in env.strategies:
        start_time = str (datetime.timedelta (seconds=k[2]))
        end_time = str (datetime.timedelta (seconds=k[3]))
        record.append ((k[0], k[1], [start_time, end_time]))
    print(len(record))

    for m in m_keys:
        for j in j_keys:
                for i in record:
                    if (m, j) == (i[1], i[0]):
                                        # print (i[2], m, j)
                                        # df.append (dict (Task='Machine %s' % (m), Start='2018-12-22 %s' % (str (i[2][0])),
                                        #          Finish='2018-12-22 %s' % (str (i[2][1])), Resource='Workflow %s' % (j + 1)))
                            df.append (dict (Task=vms[m - 1], Start='2020-01-16 %s' % (str(i[2][0])),
                                                        Finish='2020-01-16 %s' % (str (i[2][1])),
                                                        Resource='Workflow %s' % (j + 1)))
    fig = ff.create_gantt (df, index_col='Resource', show_colorbar=True, group_tasks=True,
                                            showgrid_x=True,
                                            title='DQN Workflow Scheduling')
    py.plot(fig, filename='DQN_workflow_scheduling', world_readable=True)