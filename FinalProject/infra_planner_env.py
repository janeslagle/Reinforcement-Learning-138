# Team Maintenance Agents - Punna Chowdhurry, Jane Slagle, and Diana Krmzian
# Tufts CS 138 - Final Project
# infra_planner_env.py

import numpy as np

class infra_planner:
    """
    init Parameters:

    total_years - total duration of the env simulation, defaulted at 100 years

    current_year - keeps track of the current year in the simulation, incremented at every step

    max_budget - represents the total starting budget available for maintenance actions for each year, initialized at 
    100 for each year

    state_size - how many bridges we want to run the environment with, defaulted to 1

    budget - when the simulation begins, starts out with the complete budget of 10, then gets reduced by costs of the 
    actions as the simulation progresses

    state - represents the condition of the bridge as a int percentage (starts out as 50 percent). A higher value means 
    a better condition with 100 percent being perfect condition. 

    actions - provides the action space

    is_smdp - bool to indicate if env needs to support SMDPs, takes affect in step function, controls if allows variable
    length time steps for each action

    ---------------------------------------------------------------------------------------------------------------------
    Methods:

    step(self, action, action_duration) - simulates taking time step in env with the inputted action + updates state of 
    env based on the action taken

        inputs:
        action = the action to take in the time step

        action_duration = used with running env with an SMDP algorithm. SMDP allows actions to have 
        variable time step length, so this parameter accounts for that. It only takes affect when 
        is_smdp is set to True, it doesn't impact the functionality of non-SMDP algorithms that do not 
        have variable time step lengths, defaulted at 1 in case a user runs a non-SMDP algorithm with is_smdp set to True

        outputs:
        next_state = updated state of the bridge after taking that action
        reward = reward after taking that action
        done = whether episode over or not

    calculate_reward(self, condition, prev_condition) - computes reward based on the current condt + previous condt of the bridge
        inputs:
        condition = current condition of the bridge (current state of the bridge)
        prev_condition = condition of the bridge for previous time step

        outputs:
        reward = final reward value, calculated based on bridge condition, progress made and also budget

    reset(self) - reset the env to its initial state
        outputs:
        state = re-initialized state (condition of the bridge)
    """
    def __init__(self, total_years=100, state_size=1, is_smdp=False):
        self.total_years = total_years
        self.current_year = 0
        self.max_budget = 100
        self.budget = self.max_budget
        self.state_size = state_size
        self.state = np.ones(state_size) * 40
        self.actions = ['do nothing', 'maintenance', 'replace']
        self.is_smdp = is_smdp
    
    def step(self, action, action_duration=1):
        reward = 0
        done = False       #keeps track of if episode is finished or not

        #if running env w/ SMDP algorithm then use action_duration to det how long action should take
        #with non-SMDP algorithms, each action should only take 1 time step
        if self.is_smdp:
            time_step_length = action_duration
        else:
            time_step_length = 1

        #increment current yr based on the time increment have for the action (will increment by 1 year for non-SMDP and also increment
        #for however many years the SMDP actions end up taking)
        self.current_year += time_step_length

        #store copy of current state (bridge's current condt) so that can give reward if improved upon it's condt from last time
        prev_condition = self.state.copy() 

        #episode finished when have gone throuhg total_years
        if self.current_year >= self.total_years:
            done = True  

        #cost taken out of the budget associated with taking each action
        action_costs = {'do nothing': 0, 'maintenance': 2, 'replace': 5} 

        #calc how much of the budget was used from taking that action for the amount of time the action took
        action_cost = action_costs[action] * time_step_length

        #penalize if go over budget
        #if not enough budget is left to take the inputted action then penalize it with reward
        #and can't take the action so state doesn't change
        if self.budget < action_cost:
            reward -= 10 
            next_state = self.state
        else:
            self.budget -= action_cost   #otherwise if can take action then get the budget left for next time step

            # now update state based on action taking. clip each to make sure the bridge condition stays within the bounds of 0-100
            if action == 'do nothing':
                #if bridge is being neglected, deteroriate its condition by 1%, scale it by the time step length for SMDP
                next_state = np.clip(self.state * (0.99 ** time_step_length), 0, 100)
            elif action == 'maintenance':
                #if bridge is being maintained, improve its condition by 1%
                next_state = np.clip(self.state * (1.01 ** time_step_length), 0, 100)
            elif action == 'replace':
                #have replaced the bridge so its in perfect condition
                next_state = np.ones_like(self.state) * 100

            #calc reward based on prev state + new state + scale it by the action duration
            reward += self.calculate_reward(next_state, prev_condition) * time_step_length
            
        self.state = next_state  #update state
	
        return next_state, reward, done
    
    def calculate_reward(self, condition, prev_condition):
        reward = 0

        #a bridge with state greater than or equal to 80 is a bridge in "good" condition, increase the reward
        #take the mean in case state_size is greater than 1 then can account for all
        mean_condt = np.mean(condition)
        if mean_condt >= 80:
            reward += 10
        
        #a bridge with state less than or equal to 20 is in "bad" condition, so penalize the reward
        elif mean_condt <= 20:
            reward -= 10

        #use prev_condition input to add additional reward if the condt is improving!
        prev_mean_condt = np.mean(prev_condition)
        if mean_condt > prev_mean_condt:
            reward += 1    #reward for making progress

        #checks if agent overspent budget: if did then penalize it. otherwise reward it for having good budget management skills!
        if self.budget < 0:
            reward -= 5
        else:
            reward += 2

        return reward

    def reset(self):
        #re-initialize everything!
        self.current_year = 0
        self.budget = self.max_budget
        self.state = np.ones(self.state_size) * 40
        return self.state
