import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    
    '''Plotting process'''
    h=plt.plot(Xs, Ys, '.b', markersize=16, label='Non Pareto-optimal')
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y, '.r', markersize=16, label='Pareto optimal')
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    _=plt.legend(loc=3, numpoints=1)
    plt.show()



if __name__ == '__main__':
      datapoints=np.random.rand(2, 50)

      for ii in range(0, datapoints.shape[1]):
            w=datapoints[:,ii]
            fac=.6+.4*np.linalg.norm(w)
            datapoints[:,ii]=(1/fac)*w

      h=plt.plot(datapoints[0,:], datapoints[1,:], '.b', markersize=16, label='Non Pareto-optimal')
      _=plt.title('The input data', fontsize=15)
      plt.xlabel('Objective 1', fontsize=16)
      plt.ylabel('Objective 2', fontsize=16)
      plt.show()

      plot_pareto_frontier(datapoints[0,:], datapoints[1,:], maxX=True, maxY=True)
      plot_pareto_frontier(datapoints[0,:], datapoints[1,:], maxX=False, maxY=False)

      
      