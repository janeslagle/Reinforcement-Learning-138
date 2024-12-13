# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project


## StructuRL Bridge Maintenance: Leveraging Hierarchical Reinforcement Learning
[put one sentence description of our model here, also not sure if we want to include the word "hierarchial" anymore]

## Requirements
- **Python**: 3.11.5
- **Please refer to requirements.txt for the rest of the package requirements**
- **You must run the code in the virtual environment described directly below**

## Getting Started

(1) To get started, download and unzip the code package [input whatever we name our final zip file here]

(2) Access the project folder, [input whatever we call that folder here], via your terminal

(3) Create a virtual environment

```{.py}
python -m venv .venv
```

(4) Activate the environment

```{.py}
source .venv/bin/activate
```

(5) Install the required packages by using the pip command in terminal

```{.py}
 pip install -r requirements.txt
```

## To Run and view results

(1) [will put at end when we for sure finalize EVERYTHING in our main.py file]

## Code Overview

**infra_planner.py**:
a simulation environment for bridge infrastrucutre maintenance on one bridge over a 100 year period. Budget constraints are incoporated with each action having a fixed associated cost. When determining the reward, how well the bridge is improving over time is also considered.

 State Space:
 - condition of the bridge represented as an integer value, ranging between 0 and 100 where 100 is a perfect condition and 0 is the worst possible condition
 - the condition of the bridge is always initialized as 40 so that we may observe how our model either improves or worsens the condition based on the actions it chooses to take

Action Space:
3 possible actions related to bridge infrastructure mainteance tasks:
- do nothing
- maintenance
- replace

[could briefly describe step + reward function too and then I think this is more than enough]
    
**smdp.py**
Q-learning based SMDP algorithm representing an agent that is able to interact with the infra_planner environment.

**[sarsa file]**

**[deep learning file]**

**main.py**

