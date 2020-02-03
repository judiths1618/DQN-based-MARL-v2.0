import numpy as np
import pandas as pd

from workflows.workflow import Workflow
from workflows.XMLProcess import XMLtoDAG




record = 0

class Env:
    def __init__(self, n_vm, n_agent):
        self.n_vm = n_vm
        self.n_actions = self.n_vm
        # TODO(hang): the first info can be task_id
        self.n_features = 1 + 10  # task_type and vm_state
        self.n_task = 138
        self.dim_state = self.n_task  # equal to task_num
        self.n_agent = n_agent

        self.workflow = None
        self.curr_task = None
        self.vm_time = None  # the maximum finish time for each vm
        self.vm_cost = None  # the total cost for each vm
        self.released = None  # record the released nodes of a workflow
        self.start_time = None  # record the earliest start time of a task
        self.strategies = None  # record the planning strategies
        self.state = None  # record which task has been scheduled
        self.done = None  # whether scheduling is over
        # self.reward = None
        
        N = [XMLtoDAG('.//workflows//Sipht_29.xml', 29).jobs(), XMLtoDAG('.//workflows//Montage_25.xml', 25).jobs(), XMLtoDAG('.//workflows//Inspiral_30.xml', 30).jobs(), XMLtoDAG('.//workflows//Epigenomics_24.xml', 24).jobs(), XMLtoDAG('.//workflows//CyberShake_30.xml', 30).jobs()]

        # TODO(Yuandou): 读取时间数据以及花费数据的文件
        time_tmp = pd.read_excel(".//data//WSP_dataset.xlsx", sheet_name="Containers Performance", index_col=[0])
        cost_tmp = pd.read_excel(".//data//WSP_dataset.xlsx", sheet_name="Containers Performance", index_col=[0])

        # TODO(Yuandou): 时间数据     
        timematrix = [list(map(float, time_tmp.iloc[i])) for i in range(self.n_vm)]
        costmatrix = [list(map(float, cost_tmp.iloc[i])) for i in range(self.n_vm)]
        
        global time_reward_matrix
        global cost_reward_matrix
        time_reward_matrix = timematrix
        cost_reward_matrix = costmatrix
        print(time_reward_matrix, '\n', cost_reward_matrix)
        
        self.reset()

    def reset(self):
        self.workflow = [Workflow(i) for i in range(5)]
        self.vm_time = np.zeros(self.n_vm)
        self.vm_cost = np.zeros(self.n_vm)
        self.released = [[], [], [], [], []]
        self.start_time = np.zeros(self.n_task)
        self.strategies = []
        self.state = np.ones(self.n_task, int)
        base = 0
        for i in range(len(self.workflow)):
            if i != 0:
                base += self.workflow[i - 1].size
            for j in range(len(self.workflow[i].precursor)):
                # print(self.workflow[i].precursor[j])
                idle = base + self.workflow[i].precursor[j]
                self.state[idle] = 0
        # print(self.state)

        cnt = 0
        for i in range(self.dim_state):
            if self.state[i] == 0:
                cnt += 1
        if cnt == 1 or cnt == 0:
            index = 0
        else:
            index = np.random.randint(cnt - 1)
        for i in range(self.dim_state):
            if self.state[i] == 0 and index != 0:
                index -= 1
            elif self.state[i] == 0 and index == 0:
                self.curr_task = i
                break

        self.done = False
        # self.reward = 0
        obs = []
        for i in range(self.n_agent):
            obs.append(self.observation(i))
        return obs

    def step(self, action):
        obs = []
        reward = []
        done = []

        self.set_action()
        # print('step')
        # reward.append(self.rewards(action, 0))
        # reward.append(reward[0])
        for i in range(self.n_agent):
            reward.append(self.rewards(action, i))
            obs.append(self.observation(i))
            done.append(self.is_done())
        return obs, reward, done

    @staticmethod
    def has_value(arry, value):
        for i in range(len(arry)):
            if arry[i] == value:
                return True
        return False

    def release_node(self, task):
        # print(col)
        release = []
        count = 0
        belong = []

        for i in range(len(self.workflow)):
            if task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(task - count)
                break
            count += self.workflow[i].size

        # print(belong)
        # print(self.scenario.workflows[belong[0]].structure)
        # print(self.scenario.node)
        self.released[belong[0]].append(belong[1])
        # print(self.scenario.node)

        back_node = []
        for i in range(self.workflow[belong[0]].size):
            if self.workflow[belong[0]].structure[belong[1]][i] == 1:
                back_node.append(i)

        # print(back_node)

        for i in range(len(back_node)):
            for j in range(self.workflow[belong[0]].size):
                if self.workflow[belong[0]].structure[j][back_node[i]] == 1 and not self.has_value(
                        self.released[belong[0]], j):
                    break
                elif j == self.workflow[belong[0]].size - 1:
                    release.append([belong[0], back_node[i]])
        # print(release)
        return release

    def set_action(self):
        self.state[self.curr_task] = 1

        release = self.release_node(self.curr_task)

        # print(release)
        if len(release) != 0:
            # cnt = 0
            for i in range(len(release)):
                cnt = 0
                if release[i][0] != 0:
                    for j in range(release[i][0]):
                        cnt += self.workflow[j].size
                cnt += release[i][1]
                # for i in range(7):
                self.state[cnt] = 0

        # for i in range(len(self.dim_state)):
        #     if self.state[i] == 0:
        #         self.task = i
        #         break

    def observation(self, flag):
        # get task_type
        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.curr_task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.curr_task - count)
                break
            count += self.workflow[i].size
        # print(belong)
        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        # TODO(hang): task_type can be self.curr_task
        if flag == 0:
            return np.concatenate(([task_type], self.vm_time), 0)
        else:
            return np.concatenate(([task_type], self.vm_cost), 0)
        # if flag == 0:
        #     return np.concatenate(([self.curr_task], self.vm_time), 0)
        # else:
        #     return np.concatenate(([self.curr_task], self.vm_cost), 0)

    def time_reward(self, action):
        # TODO(hang): design a smarter strategy
        strategy = []
        last_makespan = max(self.vm_time)

        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.curr_task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.curr_task - count)
                break
            count += self.workflow[i].size

        strategy.append(belong[0])
        strategy.append(action + 1)

        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        # print(agent.state.pos, type)
        exec_time = time_reward_matrix[action][task_type]
        if self.vm_time[action] >= self.start_time[self.curr_task]:
            strategy.append(self.vm_time[action])
            self.vm_time[action] += exec_time
            strategy.append(self.vm_time[action])
        else:
            strategy.append(self.start_time[self.curr_task])
            self.vm_time[action] = self.start_time[self.curr_task] + exec_time
            strategy.append(self.vm_time[action])
        # self.vm_time[action] += exec_time
        # print(vm_finish_time)
        # time = max(self.vm_time) - last_makespan

        # for i in range(self.dim_state):
        #     if self.state[i] == 0:
        #         self.task = i
        #         break

        self.strategies.append(strategy)

        finish_time = self.vm_time[action]

        back_node = []
        for i in range(self.workflow[belong[0]].size):
            if self.workflow[belong[0]].structure[belong[1]][i] == 1:
                back_node.append(i)

        for i in range(len(back_node)):
            if finish_time > self.start_time[back_node[i]]:
                self.start_time[back_node[i]] = finish_time

        # return max(vm_finish_time)
        return last_makespan, exec_time
        # return pow((4.979 - reward) / 4.079, 2)

    def cost_reward(self, action):
        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.curr_task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.curr_task - count)
                break
            count += self.workflow[i].size
        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        col = np.array(cost_reward_matrix)[:, task_type]
        worst = max(col)
        best = min(col)
        cost = col[action]
        self.vm_cost[action] += round(cost / 3600, 6)

        cnt = 0
        for i in range(self.dim_state):
            if self.state[i] == 0:
                cnt += 1
        if cnt == 1 or cnt == 0:
            index = 0
        else:
            index = np.random.randint(cnt - 1)
        for i in range(self.dim_state):
            if self.state[i] == 0 and index != 0:
                index -= 1
            elif self.state[i] == 0 and index == 0:
                self.curr_task = i
                break
        return best, worst, cost

    def rewards(self, action, flag):
        # last_makespan, exec_time = self.time_reward(action)
        # inc_makespan = max(self.vm_time) - last_makespan
        # b_cost, w_cost, a_cost = self.cost_reward(action)
        # return (pow((exec_time - inc_makespan) / exec_time, 3) + pow((w_cost - a_cost) / (w_cost - b_cost), 3)) / 2
        global record
        if flag == 0:
            last_makespan, exec_time = self.time_reward(action)
            inc_makespan = max(self.vm_time) - last_makespan
            # round(inc_makespan, 4)

            record = pow((exec_time - inc_makespan) / exec_time, 3)
            # print(record)
            return record
            # return pow((exec_time - inc_makespan) / exec_time, 3)
        else:
            b_cost, w_cost, a_cost = self.cost_reward(action)

            # return record
            return 0.2 * pow((w_cost - a_cost) / (w_cost - b_cost), 3) + 0.8 * record

    def is_done(self):
        for i in self.state:
            if i != 1:
                return False
        return True

    def compute(self, action):
        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.curr_task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.curr_task - count)
                break
            count += self.workflow[i].size
        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        last_makespan = max(self.vm_time)
        exec_time = time_reward_matrix[action][task_type]
        temp = self.vm_time[action]
        if self.vm_time[action] >= self.start_time[self.curr_task]:
            self.vm_time[action] += exec_time
            inc_makespan = max(self.vm_time) - last_makespan
        else:
            self.vm_time[action] = self.start_time[self.curr_task] + exec_time
            inc_makespan = max(self.vm_time) - last_makespan
        self.vm_time[action] = temp
        inc_makespan = round(inc_makespan, 4)

        # col = np.array(cost_reward_matrix)[:, task_type]
        # worst = col[np.argmax(col)]
        # best = col[np.argmin(col)]
        # cost = cost_reward_matrix[action][task_type]

        return pow((exec_time - inc_makespan) / exec_time, 3)