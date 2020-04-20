from workflows.XMLProcess import XMLtoDAG

import pandas as pd
import numpy as np
import datetime
import chart_studio.plotly as py
import plotly.figure_factory as ff

a = [[0, 2, 0.0, 1.522961], [0, 2, 1.522961, 3.045922], [2, 2, 3.045922, 3.265126], [2, 2, 3.265126, 3.48433], [0, 2, 3.48433, 5.007291], [0, 2, 5.007291, 6.530252000000001], [2, 2, 
6.530252000000001, 8.053213000000001], [0, 2, 8.053213000000001, 42.311885], [1, 1, 0.0, 0.248874], [2, 1, 0.248874, 2.123551], [0, 1, 2.123551, 3.998228], [2, 1, 3.998228, 4.247102], [4, 1, 4.247102, 985.988793], [4, 9, 0.0, 9.059083], [1, 9, 9.059083, 9.224563999999999], [0, 9, 985.988793, 986.874915], [4, 7, 0.0, 12.53842], [3, 7, 12.53842, 12.711319], [0, 7, 12.711319, 13.685369999999999], [4, 7, 13.685369999999999, 25.49591], [3, 7, 25.49591, 26.469960999999998], [0, 7, 26.469960999999998, 27.444011999999997], [2, 7, 27.444011999999997, 28.418062999999997], [2, 7, 28.418062999999997, 28.590961999999998], [0, 7, 28.590961999999998, 274.561878], [4, 7, 274.561878, 287.10029799999995], [2, 7, 287.10029799999995, 287.2731969999999], [0, 
7, 985.988793, 986.962844], [1, 2, 42.311885, 42.531088999999994], [0, 2, 985.988793, 987.511754], [1, 5, 0.0, 1.098401], [0, 5, 1.098401, 2.196802], [3, 5, 2.196802, 3.295203], [0, 5, 9.224563999999999, 10.322965], [3, 5, 10.322965, 357.882626], [0, 5, 357.882626, 358.98102700000004], [0, 5, 358.98102700000004, 360.07942800000006], [3, 5, 360.07942800000006, 361.1778290000001], [4, 5, 361.1778290000001, 378.6367750000001], [2, 5, 378.6367750000001, 378.8201640000001], [1, 5, 378.8201640000001, 379.9185650000001], [1, 5, 379.9185650000001, 380.1019540000001], [0, 5, 380.1019540000001, 381.2003550000001], [0, 5, 381.2003550000001, 382.29875600000014], [0, 5, 382.29875600000014, 1272.007129], [1, 6, 0.0, 1.098401], [3, 6, 1.098401, 2.196802], [4, 6, 2.196802, 19.655748000000003], [4, 6, 19.655748000000003, 36.071276], [4, 6, 36.071276, 52.48680399999999], [3, 5, 1272.007129, 1273.10553], [1, 5, 1273.10553, 1274.203931], [2, 5, 1274.203931, 1275.302332], [3, 5, 1275.302332, 1622.861993], [2, 6, 52.48680399999999, 53.585204999999995], [1, 6, 53.585204999999995, 54.683606], [3, 6, 54.683606, 72.142552], [3, 6, 72.142552, 419.70221300000003], [3, 5, 1622.861993, 1640.320939], [3, 5, 1640.320939, 1656.736467], [1, 5, 1656.736467, 1657.834868], [0, 5, 1657.834868, 1658.9332689999999], [3, 5, 1658.9332689999999, 1675.3487969999999], [0, 5, 1675.3487969999999, 1675.532186], [2, 5, 1675.532186, 1676.6305869999999], [2, 5, 1676.6305869999999, 1676.813976], [3, 5, 1676.813976, 2024.373637], [3, 6, 419.70221300000003, 437.16115900000005], [3, 6, 437.16115900000005, 454.6201050000001], [3, 6, 454.6201050000001, 471.0356330000001], [4, 6, 471.0356330000001, 487.45116100000007], [4, 6, 487.45116100000007, 503.86668900000006], [3, 6, 503.86668900000006, 520.2822170000001], [1, 5, 2024.373637, 2024.557026], [0, 5, 2024.557026, 2040.972554], [0, 6, 2040.972554, 2221.132019], [2, 4, 0.0, 1.164081], [3, 4, 1.164081, 402.38095300000003], [0, 4, 2221.132019, 2843.750419], [3, 3, 0.0, 24.41762], [1, 3, 24.41762, 25.691879999999998], [0, 7, 2221.132019, 2222.1185], [0, 3, 2221.132019, 2224.264189], [1, 3, 2224.264189, 2225.538449], [3, 3, 2225.538449, 2248.466414], [2, 3, 2248.466414, 2739.694255], [2, 5, 2040.972554, 2058.4315], 
[2, 4, 2843.750419, 2863.808291], [2, 9, 986.874915, 995.933998], [2, 9, 995.933998, 1004.993081], [2, 9, 1004.993081, 1005.879203], [0, 9, 2221.132019, 2221.8800220000003], [2, 2, 987.511754, 989.034715], [2, 2, 989.034715, 1023.293387], [2, 2, 1023.293387, 1057.552059], [3, 2, 1057.552059, 2813.140899], [2, 2, 2813.140899, 2814.66386], [2, 2, 2814.66386, 2816.1868210000002], [2, 2, 2816.1868210000002, 2817.7097820000004], [2, 2, 2817.7097820000004, 2819.2327430000005], [3, 2, 2819.2327430000005, 3173.0695830000004], [1, 1, 985.988793, 987.86347], [0, 2, 3173.0695830000004, 3196.7521480000005], [3, 1, 987.86347, 989.891174], [1, 1, 989.891174, 1971.632865], [1, 3, 2739.694255, 2764.111875], [2, 3, 2764.111875, 2788.529495], [1, 3, 2788.529495, 2811.45746], [1, 3, 2811.45746, 2834.385425], [1, 3, 2834.385425, 2857.31339], [1, 3, 2857.31339, 2880.2413549999997], [1, 3, 2880.2413549999997, 2903.1693199999995], [1, 3, 2903.1693199999995, 4151.537125999999], [0, 3, 4151.537125999999, 4152.996453999999], [1, 3, 4152.996453999999, 4405.095442999999], [1, 3, 4405.095442999999, 4406.345896999999], [1, 8, 0.0, 1.195933], [2, 3, 4406.345896999999, 4407.620156999999], [2, 1, 1971.632865, 2953.3745559999998], [4, 3, 4407.620156999999, 4898.847997999999], [4, 3, 4898.847997999999, 4923.265617999999], 
[4, 8, 1.195933, 10.255016], [4, 3, 4923.265617999999, 4947.683238], [4, 3, 4947.683238, 4972.100858], [4, 3, 4972.100858, 4996.518478], [4, 3, 4996.518478, 5020.936098], [4, 3, 5020.936098, 5043.864063], [4, 3, 5043.864063, 5066.792028], [4, 3, 5066.792028, 5089.719993], [4, 3, 5089.719993, 5112.647958], [4, 3, 5112.647958, 5135.575922999999], [4, 3, 5135.575922999999, 5159.993543], [4, 3, 5159.993543, 5182.921507999999], [4, 3, 5182.921507999999, 5205.849472999999], [4, 3, 5205.849472999999, 5230.2670929999995], [4, 3, 5230.2670929999995, 5231.541353], 
[4, 3, 5231.541353, 5254.4693179999995], [4, 3, 5254.4693179999995, 5254.667541999999]]

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
for k in a:
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

