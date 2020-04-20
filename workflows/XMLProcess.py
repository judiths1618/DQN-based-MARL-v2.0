from xml.etree.ElementTree import ElementTree
import numpy as np
import re
import xml.dom.minidom


class XMLtoDAG:
    """解析单个的scientific workflow：【id】【name】【namespace】【runtime】【transtime】"""
    job_tag = "{http://pegasus.isi.edu/schema/DAX}job"
    child_tag = "{http://pegasus.isi.edu/schema/DAX}child"
    parent_tag = "{http://pegasus.isi.edu/schema/DAX}parent"
    uses_tag = "{http://pegasus.isi.edu/schema/DAX}uses"

    def __init__(self, file, n_task):
        # 初始化
        self.xmlFile = file
        self.n_task = n_task
        self.DAG = np.zeros((self.n_task, self.n_task), dtype=int)
        self.taskType = np.zeros((self.n_task), dtype=int)  # workflow中的task类型

    def get_dag(self):
        # 使用minidom解析器打开 XML 文档
        domtree = xml.dom.minidom.parse(self.xmlFile)
        collection = domtree.documentElement
        childrens = collection.getElementsByTagName("child")

        for child in childrens:
            child_id = child.getAttribute('ref')
            child_id = int(child_id[2:])
            # print('Child: ', child_id)
            parents = child.getElementsByTagName('parent')
            for parent in parents:
                parent_id = parent.getAttribute('ref')
                parent_id = int(parent_id[2:])
                # print(parent_id)
                self.DAG[parent_id, child_id] = 1
        return self.DAG

    def get_precursor(self):
        # 获得任务的前驱结点集合
        precursor = []
        for i in range(self.n_task):
            temp = self.DAG[:, i]
            if np.sum(temp) == 0:
                precursor.append(i)
        return precursor

    def print_graph(self):
        print(self.DAG)
        for i in range(self.n_task):
            for j in range(self.n_task):
                if self.DAG[i, j] != 0:
                    print(i, ' -> ', j)

    def jobs(self):  # 所有的job任务的集合
        tree = ElementTree(file=self.xmlFile)
        root = tree.getroot()
        simple_jobs = []
        for job in root.iter(tag=self.job_tag):
            input_speed = []
            input_size = []
            # print(len(job.findall(self.uses_tag)))
            for use in job.findall(self.uses_tag):
                if use.get('link')=='input':
                    use_file_size = int(use.get('size'))    # the unit of size: B
                    input_speed.append(round(use_file_size/(10*1024*1024),6))    # Network I/O speed=10M/s
                    input_size.append(use_file_size)
            simple_job = {'id': job.attrib['id'], 
                            'name': job.attrib['name'], 
                            'namespace': job.attrib['namespace'],
                            'runtime': float(job.attrib['runtime']),    
                             # 'transtime': float(round(max(input_speed),6)),    # 考虑并行传输文件
                            'transtime': float(round(sum(input_size)/(10*1024*1024),6)),   # 考虑串行传输文件
                            'minsize': float(round(sum(input_size)/(1024*1024*1024),6))    # Minimum Storage size:  * GB
                          }
            simple_jobs.append(simple_job)
        return simple_jobs

    def types(self):  # 所有任务的类型
        types = []
        res = []
        for job in self.jobs(): 
            types.append(job['name'])
        for i, type in enumerate({}.fromkeys(types).keys()):
            res.append(type)
        for i, type in enumerate(types):
            self.taskType[i]=res.index(type)
            # print(self.taskType[i])
        return res, self.taskType

    def typeRTimeDicts(self, types, jobs):  # 每种任务类型对应的执行时间的集合
        typeRTimeDict = {}
        for j, typ in enumerate(types):
            lst = []
            for i, job in enumerate(jobs):
                if job['name'] == typ:
                    lst.append(job['runtime'])
            print(typ, lst)
            typeRTimeDict[typ] = lst
        return typeRTimeDict

    def typeTTimeDicts(self, types, jobs):  # 每种任务类型对应的传输时间的集合
        typTTimeDict = {}
        for j, typ in enumerate(types):
            lst = []
            for i, job in enumerate(jobs):
                if job['name'] == typ:
                    lst.append(job['transtime'])
            typTTimeDict[typ] = lst
        return typTTimeDict


if __name__ == '__main__':
    import csv
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import os
    print(os.path.abspath('.'))
    WFS = ['Sipht_29.xml', 'Montage_25.xml', 'Inspiral_30.xml', 'Epigenomics_24.xml',
        'CyberShake_30.xml']
    N = [29, 25, 30, 24, 30]
    WLPath = ['./workflows/'+wl for wl in WFS]

    temps = []
    Jobs = []
    TYPES = []
    TASK_TYPES = []
    for wf, n in zip(WLPath, N):
        temps.append(XMLtoDAG(wf, n))
        Jobs.append(XMLtoDAG(wf, n).jobs())
        TYPES += XMLtoDAG(wf, n).types()[0]
        TASK_TYPES.append(XMLtoDAG(wf, n).types()[1])
     
    c_tmp = pd.read_excel(".//data//WSP_dataset.xlsx", sheet_name="Containers Price")
    CONTAINERS = list(c_tmp.loc[:, 'Configuration Types'])
    PRF = list(c_tmp.loc[:, 'Performance Ref'])     # the higher (prf), the faster (runtime).
    CST = list(c_tmp.loc[:, 'Cost'])     # the higher (prf), the faster (runtime).
    print(CONTAINERS, PRF, CST)

    print(len(TYPES[0]), TYPES[0])
    print(len(TASK_TYPES), TASK_TYPES)
    # TODO: 写文件 service time dataset     sertime=np.mean(runtime)/prf+np.mean(transtime)
    mymat = np.zeros((len(PRF), 39), dtype=float)
    costmat = np.zeros((len(CST), 39), dtype=float)
    j = -1
    for prf, cst in zip(PRF, CST):
        cnt = -1
        j += 1 
        # print(j, prf, cst)
        for k, graph in enumerate(temps):
            jobs = graph.jobs()
            types = graph.types()[0]
            rt = graph.typeRTimeDicts(types, jobs)
            tt = graph.typeTTimeDicts(types, jobs)
            
            for i, typ in enumerate(types):
                cnt += 1
                arv_rt = np.mean(list(rt.values())[i])
                arv_tt = np.mean(list(tt.values())[i])                
                srt = arv_rt/prf + arv_tt
                cost = srt* cst
                mymat[j][cnt] = round(srt, 6)
                costmat[j][cnt] = round(cost, 6)
                # print(cnt, j, i, '\t', prf, typ, '\t', srt, mymat[j][cnt])
    print(costmat)
    df = pd.DataFrame(costmat)
    # df.to_csv('.//data//Org_dataset-1.csv')

    for wf, n, wl in zip(WLPath, N, WFS):
        # ------ plot the features of runtime and transmission time for each workflow -------
        graph = XMLtoDAG(wf, n)
        jobs = graph.jobs()
        types = graph.types()[0]
        rt = graph.typeRTimeDicts(types, jobs)
        tt = graph.typeTTimeDicts(types, jobs)
        labels = [str(i+1) for i in range(len(list(rt.keys())))]
        runtime = list(rt.values())
        transtime = list(tt.values())
        fs = 10         # fontsize

        # print(transtime)
        plt.boxplot(runtime, labels=labels, showmeans=True, meanline=True)
        plt.ylabel('runtime (s)')
        plt.xlabel('taskType #')
        plt.title(wf)
        plt.savefig('./figures/time_features/rt_'+wl+'.svg')
        # plt.show()

        plt.boxplot(transtime, labels=labels, showmeans=True, meanline=True)
        plt.ylabel('transmission time (s)')
        plt.xlabel('taskType #')
        plt.title(wf)
        plt.savefig('./figures/time_features/tt_'+wl+'.svg')
        # plt.show()


