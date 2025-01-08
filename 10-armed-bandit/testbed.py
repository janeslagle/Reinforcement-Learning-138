# Jane Slagle
# CS 138 - Programming Assignment 1
# testbed.py

import numpy as np

class TestBed:
    '''
    Simulation of non-stat multi-armed bandit prob, specifically 10-armed testbed
    This class handles randomly updating vals of each action through random walk, calcs reward for each action at each time step based on normal distribution
    '''
    def __init__(self):
        self.num_arms = 10                                   # 10-armed bandit
        self.std_dev = 0.01                                  # standard dev for random walks, told in prob statement
        self.q_star = np.random.normal(0, 1, self.num_arms)  # init all rewards from normal distrib w/ mean 0, variance 1 (Section 2.3 of book)

    def random_walk(self):
        '''
        Take independent random walks by adding normally distributed increment w/ mean 0, std dev 0.01 to reward at each time step
        '''
        self.q_star += np.random.normal(0, self.std_dev, self.num_arms)

    def reward_val(self, act):
        '''
        R_t is given by normal distribution with mean equal to true reward val of current action, variance 1 as stated in Section 2.3 of book 
        elements of q_*(A_t) are mean values of each of 10 arms in prob so we want each arm as input
        '''
        return np.random.normal(self.q_star[act], 1)
