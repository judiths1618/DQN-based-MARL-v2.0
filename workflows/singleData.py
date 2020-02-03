from XMLProcess import XMLtoDAG
import matplotlib.pyplot as plt
import numpy as np

f = "Sipht_968.xml"
wl = XMLtoDAG('.//workflows//'+f, 968)


jobs = wl.jobs()
types = wl.types()

dic1 = wl.typeRTimeDicts(types, jobs)
dic2 = wl.typeTTimeDicts(types, jobs)

print(list(dic1.keys()))

labels = [str(i+1) for i in range(len(list(dic1.keys())))]
runtime = list(dic1.values())
transtime = list(dic2.values())
fs = 10         # fontsize

print(transtime)
plt.boxplot(runtime, labels=labels, showmeans=True, meanline=True)
plt.ylabel('runtime (s)')
plt.xlabel('taskType #')
plt.title(f)
plt.show()

plt.boxplot(transtime, labels=labels, showmeans=True, meanline=True)
plt.ylabel('transmission time (s)')
plt.xlabel('taskType #')
plt.title(f)
plt.show()

# x_l = [i for i, task in enumerate(jobs)]
# sertime = [float(i['sertime']) for i in jobs]
# fig, ax = plt.subplots()
# ax.plot(x_l, sertime, 'ko-')

# ax.set(xlabel='taskId', ylabel='service time (s)',
#        title='the service time of single task')
# ax.grid()

# fig.savefig("test.png")
# plt.show()




