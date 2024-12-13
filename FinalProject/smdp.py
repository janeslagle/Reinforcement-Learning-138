# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# smdp.py

import numpy as np
import random

class SMDP:
    """
    env - infra_planner env that this agent interacts with!

    num_eps - num episodes to run the algorithm

    num_acts = number of actions that agent is able to take from the env, want for init Q table with 
    q_vals param

    q_vals - stores all q values (expected culmul. reward) for each action, init as 1D array of all 0s.
    can make 1D here bc only one state (the env is only for 1 bridge), so the Q table here just needs
    to store all poss actions for that one state we have!

    all_acts_taken - keep track of all of the actions that the agent takes when run through SMDP
    so that can use it in our results, init as empty list

    eps, alph, gamma - parameters for formulas, initialized as their usual values used with RL tasks

    ----------------------------------------------------------------------------------------------------
    Methods:
    
    choose_act(self, state) - choose action for inputted state using eps-greedy policy approach
        inputs:
        state = the state from env for which we are taking an action for

        outputs:
        action = the best action to take for the inputted state

    choose_rand_act(self) - helper func for choose_act, randomly selects an act (explores) in eps-greedy 
    act selection approach
    
    choose_best_act(self) - helper func for choose_act, selects act w/ highest est value (exploits)

    calc_q_val(self, action, reward, action_duration) - calculates the q value for the inputted action 
    based on the normal updated Q(s,a) func where Q(s,a) += alpha * [reward at current action + gamma * q value 
    for action that gives max t that state - current act q value]
        
        outputs:
        none, simply just updates the q_vals for the action just took

    run_smdp - simulates running the SMDP algor over all the inputted episodes with the env
        outputs:
        rewards = rewards for each episode
    """
    def __init__(self, env, num_eps):
        self.env = env
        self.num_eps = num_eps
        self.num_acts = len(env.actions)
        self.q_vals = np.zeros(self.num_acts)
        self.all_acts_taken = []
        self.eps = 0.1
        self.alph = 0.1
        self.gamma = 0.9

    def choose_act(self):
        #explore w/ prob eps = means take rand act
        if np.random.uniform() < self.eps:
            #get the corresp action out of our list of actions from the env
            action = self.env.actions[self.choose_rand_act()]
        else:
            action = self.env.actions[self.choose_best_act()]
        
        #now that have taken this action, need to keep track that we took it for our results by adding it to our all_acts_taken list!!
        self.all_acts_taken.append(action)

        return action

    def choose_rand_act(self):
        return random.choice(range(self.num_acts))  
        
    def choose_best_act(self):
        return np.argmax(self.q_vals)

    def calc_q_val(self, action, reward, action_duration):
        #calc based off equation specified in docstrings above
        act = self.env.actions.index(action)  #get out which specific act from all poss acts are taking here so that can udpate the q val for it

        #need scale the discount factor by action_duration since time spent on action will impact the q val to accurately depict how much time spent
        #on action should discount its q val
        self.q_vals[act] += self.alph * (reward + (self.gamma ** action_duration) * (np.max(self.q_vals) - self.q_vals[act]))

    def run_smdp(self, num_episodes):
        rewards = []   #keep track of total reward so that can use it for results

        for eps in range(num_episodes):
            self.env.reset()              #reset env at start of each episode
            total_reward = 0     
            done = False

            while not done:
                act = self.choose_act()   #simulate taking act

                #need specify the action duration window to use in env step func
                #choose a random num btw 1,5 years to simulate real world randomness
                if self.env.is_smdp:
                    action_duration = np.random.randint(1, 6)
                else:
                    action_duration = 1

                next_state, reward, done = self.env.step(act, action_duration)  #simulate taking step in env w/ action that has action duration just found
                self.calc_q_val(act, reward, action_duration)                   #simulate updating the q val for taking that action
                total_reward += reward 

            rewards.append(total_reward)

            #print out when make it to every 1,000th episode so that know that the algorithm is actually running + working when test it + get results out
            if eps % 1000 == 0:
               print("Episode " + str(eps))

        return rewards 
