# Jane Slagle
# CS 138 - Programming Assignment 1
# agent.py

import numpy as np
import random

class Agent:
    '''
    Simulate agent that will interact with 10-armed bandit env
    Agent will use eps-greedy policy or UCB to select actions, update act-value ests
    '''
    def __init__(self, alph=None):
        self.num_arms = 10                                   # fixed 10-armed testbed
        self.eps = 0.1                                       # fixed epsilon = 0.1 epsilon-greedy approach
        self.alph = alph                                     # will either be None if user does not specify or 0.1 if user does specify
        self.q_ests = np.zeros(self.num_arms)                # act value estimates
        self.num_times_act_chosen = np.zeros(self.num_arms)  # keep count of number times each action chosen when updating act value ests
        self.c = 2                                           # c used in UCB approach
        self.time_step = 0                                   # need t in UCB approach where t = current time step on in run through experiment 

    def eps_greedy_selection(self):
        '''
        Choose next action based on eps-greedy approach
        Generate rand num btw 0, 1 to compare against eps value to make exploration/exploitation decision for act selection
        '''
        if np.random.uniform() < self.eps:
            return self.choose_rand_act()
        else:
            return self.choose_best_act()
            
    def choose_rand_act(self):
        '''
        Randomly selects an act (explores) in eps-greedy act selection approach
        '''
        return random.choice(range(self.num_arms))  
        
    def choose_best_act(self):
        '''
        Selects act w/ highest est value (exploits)
        '''
        return np.argmax(self.q_ests)
        
    def ucb_selection(self):
        '''
        Choose next action based on UCB approach
        Use eq 2.10 given in book for A_t
        if N_t(a) = 0 then act a = max act to use there
        '''
        self.time_step += 1              # increment time step on each time call method
        ucbs = np.zeros(self.num_arms)   # want keep track of all values get from eq 2.10 so that can take, return argmax at end
        
        # want loop over all acts have
        for act in range(self.num_arms):
            # if num times chosen act = 0 then return that action as one to select
            if self.num_times_act_chosen[act] == 0:
                return act
            else:
                # eq 2.10 from book
                ucbs[act] = self.q_ests[act] + (self.c * np.sqrt(np.log(self.time_step) / self.num_times_act_chosen[act]))
        
        # want return argmax of ucb values found from eq 2.10
        return np.argmax(ucbs)
        
    def act_val_method(self, act, reward):
        '''
        Update act-value ests based on chosen act, its reward
        '''
        # increment act count 1st so that don't get divide by 0 err
        self.num_times_act_chosen[act] += 1

        if self.alph is None:
            # incremental sample-avg method case
            alph_step_size = 1 / self.num_times_act_chosen[act] 
            self.q_ests[act] += alph_step_size * (reward - self.q_ests[act])
        else:
            # constant step size case, use self.alph as step size
            self.q_ests[act] += self.alph * (reward - self.q_ests[act])
