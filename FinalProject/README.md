# Team Maintenance Agents
## Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project

## Requirements
- **Python**: 3.11.5
- **Please refer to requirements.txt for the rest of the package requirements**
- **You must run the code in the virtual environment described directly below**

## StructuRL Bridge Maintenance: Leveraging Hierarchical Reinforcement Learning
[put one sentence description of our model here, also not sure if we want to include the word "hierarchial anymore"]

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

(1)

## Code Overview

**infra_planner.py**



    A simulation environment for bridge infrastructure maintenance
    Represents a simulation environment for bridge infrastructure maintenance. We simulate making repairs on 
    one bridge. We simulate taking maintenance with one bridge over a 100 year period.
    State space = condition of one bridge represented as an integer value. 100 = perfect condition, 0 = worst possible condition
    The condition of the bridge is always initialized as 40 so that we may observe how our model either improves or worsens based
    on the actions taken.

    Action space = related to bridge infrastructure mainteance tasks. 3 possible actions: 
    (1) do  nothing
    (2) maintenance
    (3) replace

    We include budget constraints with each action having a fixed associated cost with rewards being computed based on if 
    the bridge is improving over time as well as how well the agent is maintaining the budget.

**smdp.py**

**[sarsa file]**

**[deep learning file]**

**main.py**

