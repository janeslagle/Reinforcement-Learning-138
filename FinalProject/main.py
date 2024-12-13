# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# main.py

import numpy as np
import matplotlib.pyplot as plt
from infra_planner_env import infra_planner
from smdp import SMDP

def plot_rewards(rewards, algor_name):
    #get a moving average (rolling mean) of the rewards to smooth the rewards over episodes curve
    window_size = 100 

    if len(rewards) >= window_size:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    else:
        smoothed_rewards = rewards

    plt.subplot(2, 1, 1)
    plt.plot(np.cumsum(rewards), label='Cumulative Rewards', color = "deeppink")
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Rewards')
    plt.title(str(algor_name) + ' Cumulative Rewards Over Episodes')

    plt.subplot(2, 1, 2)
    plt.plot(rewards, label='Rewards per Episode', color='lightskyblue', alpha=0.5)

    if len(smoothed_rewards) < len(rewards):
        smoothed_x = np.arange(window_size - 1, len(rewards))
    else:
        smoothed_x = np.arange(len(smoothed_rewards))

    plt.plot(smoothed_x, smoothed_rewards, label=f'Mean (window={window_size})', color='cornflowerblue', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(str(algor_name) + ' Rewards Per Episode (Original Results & Smoothed Average)')
    plt.legend()

    plt.suptitle("Culmulative and Per-Epsiode Rewards for " + str(algor_name))

    plt.tight_layout()
    plt.show()

def avg_culm_reward(rewards, algor_name):
    avg_reward = np.round(np.mean(rewards), 2)

    print("")
    print(str(algor_name) + " Average Cumulative Reward: " + str(avg_reward))

    return avg_reward

#compare rewards w/ amount of budget used
def cost_effic(env, avg_reward, algor_name):
    total_cost_spent = env.max_budget - env.budget

    cost_effic = avg_reward / total_cost_spent

    print("")
    print(str(algor_name) + " Total Cost Efficiency: " + str(np.round(cost_effic, 2)))

    return cost_effic

def each_act_num(env, agent, algor_name):
    #store amt times took each act as a dict where key = the action, value = amount of times took that action in agent when running algor
    each_act_amt = {action: agent.all_acts_taken.count(action) for action in env.actions}

    total_acts_taken = sum(each_act_amt.values())

    print("")
    print(str(algor_name) + " Percentage of times took each action out of " + str(total_acts_taken) + " times:")

    for act, num_times in each_act_amt.items():
        act_percent = (num_times / total_acts_taken) * 100
        print("Action " + str(act) + " " + str(np.round(act_percent, 2)) + "% " + "(" + str(num_times) + " times)")

    return each_act_amt

def condt_changes(init_condt, final_condt, algor_name):
    condt_percent = np.round(((final_condt - init_condt) / (100 - init_condt)) * 100, 2)

    print("")
    print(str(algor_name) + " Bridge Initial Condition: " + str(init_condt))
    print(str(algor_name) + " Bridge Final Condition: " + str(final_condt))

    #figure out if condition of bridge improved or deteroriated
    if condt_percent > 0:
        print(str(algor_name) + " Bridge Condition improved by " + str(condt_percent) + "%")
    elif condt_percent < 0:
        print(str(algor_name) + " Bridge condition deteriorated by " + str(condt_percent * -1) + "%")
    else: 
        print(str(algor_name) + " Bridge condition did not change.")

def budget_changes(init_budget, final_budget, algor_name):
    spent_budget = ((init_budget - final_budget) / init_budget) * 100
    budget_left = 100 - spent_budget
    
    print("")
    print(str(algor_name) + " Initial Budget: " + str(init_budget))
    print(str(algor_name) + " Final Budget: " + str(final_budget))
    print(str(algor_name) + " Percentage of total budget spent: " + str(spent_budget) + "%")
    print(str(algor_name) + " Percentage of total budget leftover: " + str(budget_left) + "%")

def get_smdp_results(num_episodes, get_plot_rewards, get_avg_culm_reward, get_each_act_num, get_cost_effic, get_budget_changes, get_condt_changes):
    smdp_algor_name = "SMDP"

    #1st set up env to use it with SMDP
    smdp_env = infra_planner(is_smdp=True)

    #now that have init env, get the init state + budget out of it
    init_smdp_state = int(smdp_env.reset())
    init_smdp_budget = smdp_env.max_budget

    #now set up agent
    smdp_agent = SMDP(smdp_env, num_eps=num_episodes)

    #now actually run the env + get the reward results out, yahoo!
    smdp_rewards = smdp_agent.run_algor(num_episodes=num_episodes)

    #now that have actually run algor w/ env, can get final state + budget out
    final_smdp_state = int(smdp_env.state)
    final_smdp_budget = smdp_env.budget

    #can also get the avg culm reward out now that have all the rewards
    smdp_avg_culm_reward = avg_culm_reward(smdp_rewards, smdp_algor_name)

    if get_plot_rewards:
        plot_rewards(smdp_rewards, smdp_algor_name)

    if get_avg_culm_reward:
        smdp_avg_culm_reward

    if get_each_act_num:
        each_act_num(smdp_env, smdp_agent, smdp_algor_name)

    if get_cost_effic:
        cost_effic(smdp_env, smdp_avg_culm_reward, smdp_algor_name)

    if get_budget_changes:
        budget_changes(init_smdp_budget, final_smdp_budget, smdp_algor_name)
    
    if get_condt_changes:
        condt_changes(int(init_smdp_state), final_smdp_state, smdp_algor_name)

if __name__ == '__main__':
    num_episodes = 100000   #want run all algor w/ the same number of episodes = 100000

    #inputted bools for each algor:
    #(1) whether to call plot_rewards 
    #(2) whether to call get_avg_culm_reward 
    #(3) whether to call each_act_num
    #(4) whether to call get_cost_effic
    #(5) whether to call budget_changes
    #(6) whether to call condt_changes

    #run w/ smdp
    get_smdp_results(num_episodes, True, True, True, True, True, True)

    pass
