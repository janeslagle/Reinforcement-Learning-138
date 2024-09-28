# Jane Slagle
# CS 138 - Programming Assignment 1
# main.py

import numpy as np
import matplotlib.pyplot as plt
from testbed import TestBed
from agent import Agent

def run_testbed(num_runs, num_steps, alph, isEpsGreedy):
    rewards = np.zeros((num_runs, num_steps))
    opt_acts = np.zeros((num_runs, num_steps))

    for run in range(num_runs):
        print(run)
        testbed = TestBed() 
        agent = Agent(alph=alph)

        for time_step in range(num_steps):
            if isEpsGreedy:
                act = agent.eps_greedy_selection()    # choose act
            else:
                act = agent.ucb_selection() 
            reward = testbed.reward_val(act)          # get reward
            agent.act_val_method(act, reward)         # update act-value est
            testbed.random_walk()                     # take rand walk

            rewards[run, time_step] = reward          # store reward val just got
            
            # figure out if act just got is opt act
            if act == np.argmax(testbed.q_star):
                # if yes, indicate so
                opt_acts[run, time_step] = 1
            else:
                # if not, also indicate so
                opt_acts[run, time_step] = 0

    rewards_avg = np.mean(rewards, axis=0)        # want take avg (mean) over 2000 runs, not 10000 time steps
    opt_acts_avg = np.mean(opt_acts, axis=0) * 100
    
    return rewards_avg, opt_acts_avg

def plot_eps_greedy():
    samp_avg_rewards, samp_opt_acts_avg = run_testbed(num_runs=2000, num_steps=10000, alph=None, isEpsGreedy = True)
    const_avg_rewards, const_opt_acts_avg = run_testbed(num_runs=2000, num_steps=10000, alph=0.1, isEpsGreedy = True)

    plt.subplot(2, 1, 1)
    plt.title('Epsilon-Greedy Average Rewards (epsilon = 0.1)')
    plt.plot(samp_avg_rewards, label='Sample Average (alpha=1/n)', color='cornflowerblue')
    plt.plot(const_avg_rewards, label='Constant Step-Size (alpha=0.1)', color='deeppink')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Epsilon-Greedy Optimal Action Percentage (epsilon = 0.1)')
    plt.plot(samp_opt_acts_avg, label='Sample Average (alpha=1/n)', color='cornflowerblue')
    plt.plot(const_opt_acts_avg, label='Constant Step-Size (alpha=0.1)', color='deeppink')
    plt.ylim(0, 100)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()

    plt.show()
    
def plot_UCB():
    samp_avg_rewards, samp_opt_acts_avg = run_testbed(num_runs=2000, num_steps=10000, alph=None, isEpsGreedy = False)
    const_avg_rewards, const_opt_acts_avg = run_testbed(num_runs=2000, num_steps=10000, alph=0.1, isEpsGreedy = False)

    plt.subplot(2, 1, 1)
    plt.title('UCB Average Rewards (epsilon = 0.1)')
    plt.plot(samp_avg_rewards, label='Sample Average (alpha=1/n)', color='cornflowerblue')
    plt.plot(const_avg_rewards, label='Constant Step-Size (alpha=0.1)', color='deeppink')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('UCB Optimal Action Percentage (epsilon = 0.1)')
    plt.plot(samp_opt_acts_avg, label='Sample Average (alpha=1/n)', color='cornflowerblue')
    plt.plot(const_opt_acts_avg, label='Constant Step-Size (alpha=0.1)', color='deeppink')
    plt.ylim(0, 100)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    
    plt.show()
    
if __name__ == '__main__':
    #plot_eps_greedy()
    #plot_UCB()
     
    pass
