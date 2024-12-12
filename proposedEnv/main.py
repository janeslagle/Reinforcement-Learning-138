import numpy as np
import matplotlib.pyplot as plt
from proposed_env import InfraPlanner
from smdp import SMDP

# init the env w/ SMDP, get all results with SMDP!!!
use_smdp = True
smdp_env = InfraPlanner(total_years=100, max_budget=1000, include_budget=True, state_size=10, is_smdp =use_smdp)

agent = SMDP(smdp_env, gamma=0.9, alpha=0.1, epsilon=0.1)  # plug in usual param values for gamma, alpha, epsilon
rewards = agent.train(episodes=100000)                     # train the agent!

# plot culm. rewards over all episodes  
plt.plot(np.cumsum(rewards))
plt.xlabel('Episodes')
plt.ylabel('Cumulative Rewards')
plt.title('SMDP Agent Training Performance')
plt.show()

# get out all performance metrics we want
def analyze_performance(agent):
    avg_reward = np.mean(rewards)
    print(f"Average Cumulative Reward: {avg_reward}")

    total_cost = (smdp_env.max_budget - smdp_env.budget)
    cost_efficiency = avg_reward / total_cost
    print(f"Cost Efficiency: {cost_efficiency}")

    action_counts = {action: agent.action_history.count(action) for action in agent.env.action_space}
    print(f"Action Counts (Maintenance, Replacement, Neglect): {action_counts}")

analyze_performance(agent)
