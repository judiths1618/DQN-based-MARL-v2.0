import numpy as np
from workflows.subtask import SubTask
from workflows.XMLProcess import XMLtoDAG


TYPE = ['./workflows/Sipht_29.xml', './workflows/Montage_25.xml', './workflows/Inspiral_30.xml', './workflows/Epigenomics_24.xml',
        './workflows/CyberShake_30.xml']
NUMBER = [29, 25, 30, 24, 30]

# TODO(Yuandou): 匹配具体的任务类型转换
TASK_TYPE = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2],
             [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 6, 7],
             [0, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4]]


class Workflow:
    def __init__(self, num):
        self.id = num + 1
        self.type = TYPE[num]
        self.size = NUMBER[num]
        self.subTask = [SubTask((num + 1) * 1000 + i + 1, TASK_TYPE[num][i]) for i in range(self.size)]  # 子任务

        dag = XMLtoDAG(self.type, self.size)
        self.structure = dag.get_dag()  # 带权DAG
        self.precursor = dag.get_precursor()

if __name__ == "__main__":
    wl = Workflow(0)
    st = SubTask(0, TASK_TYPE)
    print(wl.id, wl.type, wl.size, len(wl.subTask),'\n', wl.structure, '\n', wl.precursor)

    wl.structure = np.delete(wl.structure, wl.precursor, 0)
    wl.structure = np.delete(wl.structure, wl.precursor, 1)
    print(wl.structure, st)
        # print(self.precursor)
        # self.structure = np.delete(self.structure, self.precursor, 0)
        # self.structure = np.delete(self.structure, self.precursor, 1)
        # print(self.structure)
