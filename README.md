# DQN-based-MARL-v2.0

## Overview
This work is related to my paper called 'Multi-Objective Workflow Scheduling With Deep-Q-Network-Based Multi-Agent Reinforcement Learning', which proposed a novel method for scientific workflow scheduling over heterogeneous virtual machines. Based on the algorithm framework, now i am attempting to utilize the pareto frontier technology combining with the two DQN agents for a bi-objective optimization workflow scheduling problem on heterogeneous containers.

This example uses the [Pegasus Scientific Workflows] (https://confluence.pegasus.isi.edu/display/pegasus/WorkflowGenerator) as the workflow templates and the [AWS Fargate Pricing] (https://aws.amazon.com/fargate/pricing/) as the resource model. 

## Requirements
Python 3.6.0+

## Usage
To run the program, please execute the following from the root directory: 

```
conda env create -f myDQN.yaml
```
or 
```
pip install -r requirements.txt
```

Once the environment is successfully installed, then go to run the **run_this.py** file

## Some results
### Time features
#### The distribution of runtime for each task type
![CyberShake_30.xml](./figures/time_features/rt_CyberShake_30.xml.svg)
![Epigenomics_24.xml](./figures/time_features/rt_Epigenomics_24.xml.svg)
![Inspiral_30.xml](./figures/time_features/rt_Inspiral_30.xml.svg)
![Montage_25.xml](./figures/time_features/rt_Montage_25.xml.svg)
![Sipht_29.xml](./figures/time_features/rt_Sipht_29.xml.svg)

#### The distribution of transmission time for each task type
![CyberShake_30.xml](./figures/time_features/tt_CyberShake_30.xml.svg)
![Epigenomics_24.xml](./figures/time_features/tt_Epigenomics_24.xml.svg)
![Inspiral_30.xml](./figures/time_features/tt_Inspiral_30.xml.svg)
![Montage_25.xml](./figures/time_features/tt_Montage_25.xml.svg)
![Sipht_29.xml](./figures/time_features/tt_Sipht_29.xml.svg)

### Reward features
![rewards curve](./figures/reward_features/rewards.png)

### Pareto frontier
![pareto optimal](./figures/pareto_frontier/Pareto.png)

### Plans
![gantt graph](./figures/plans/DQN-based_MARL_workflow_scheduling.png)
