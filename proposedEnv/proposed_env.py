import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class InfraPlanner:
    def __init__(self, total_years=100, max_budget=1000, include_budget=True, state_size=10, is_smdp=False):
        self.action_map = {'maintenance': 1, 'replacement': 2, 'neglect': 3}  # this is for the deteoriation env simulation, just maps them to values so that we can plot them. that's all!
        self.total_years = total_years
        self.current_year = 0
        self.max_budget = max_budget
        self.budget = max_budget
        self.include_budget = include_budget
        self.state_size = state_size
        self.state = np.ones(state_size) * 100  #have bridge start out in excellent, perfect condition
        self.is_smdp = is_smdp
        
        # the action space for what we want to simulate / do with the bridges
        self.action_space = ['maintenance', 'replacement', 'neglect']
        
    # note: need figure out how use dt exactly with SMDP, but can finetune that later
    def step(self, action, dt=1):
        reward = -1  # default penalty / time step cost?
        done = False
        time_increment = dt if self.is_smdp else 1 
        self.current_year += time_increment

        if self.current_year >= self.total_years:
            done = True  # episode ends when time is up

        # cost for each action + take it out of budget
        action_costs = {'maintenance': 2, 'replacement': 5, 'neglect': 1}
        budget_used = action_costs[action] * time_increment

        # constrain with budget --> makes it more realistic I think
        if self.budget < budget_used:
            reward -= 10  # penalty if go over budget. can finetune this value later
            next_state = self.state
        else:
            self.budget -= budget_used
            # effects from taking each action on the state of the bridge. make it dependent on the current state --> think makes it more realistic
            if action == 'maintenance':
                next_state = self.state * (0.95 ** time_increment)
            elif action == 'replacement':
                next_state = np.ones_like(self.state) * 100
            elif action == 'neglect':
                next_state = self.state * (1.05 ** time_increment)

            next_state = np.clip(next_state, 0, 100)  # ensure bridge condition stays within bounds of 0-100
            reward += self.calculate_reward(next_state)

        return next_state, reward, done, {}           # have return same info that OG env did

    # calcs reward based on the bridge's condition
    def calculate_reward(self, condition):
        reward = 0   # init as 0

        # just set these values for what a "good" and "bad" condition are. Anything above 80 is counted as good, so increase the reward. anything below 20 means
        # bad so decrease the reward. We can finetune this later, just getting something for nows
        if np.mean(condition) > 80:
            reward += 10
        elif np.mean(condition) < 20:
            reward -= 10

        # figure out how it impacts the budget
        if self.include_budget:
            if self.budget < 0:
                reward -= 5
            else:
                reward += 2

        return reward

    def reset(self):
        self.current_year = 0
        self.budget = self.max_budget
        self.state = np.ones(self.state_size) * 100
        return self.state
    
    def plot_deterioration(self, episodes=100):
        # rest env
        state = self.reset()

        # init the "history" for each of these over the entire animation time duration
        years = []
        condition_history = []
        speed_history = []
        rewards = []
        actions = []

        # set up plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
        fig.suptitle('Deterioration Simulation')

        # set up the 4 subplots for the entire plot
        ax1.set(xlabel='Year', ylabel='Condition', title='Condition Over Time', ylim=(0, 100))
        ax2.set(xlabel='Year', ylabel='Speed', title='Deterioration Speed', ylim=(-10, 10))
        ax3.set(xlabel='Year', ylabel='Cumulative Rewards', title='Cumulative Rewards', ylim=(-100, 0))
        ax4.set(xlabel='Year', ylabel='Actions', title='Actions Over Time', ylim=(0, 5))

        # init plots as empty each time
        condition_line, = ax1.plot([], [], 'b-', label='Condition')
        speed_line, = ax2.plot([], [], 'r-', label='Speed')
        reward_line, = ax3.plot([], [], 'g-', label='Cumulative Reward')
        bars = ax4.bar([], [])

        def init():
            condition_line.set_data([], [])
            speed_line.set_data([], [])
            reward_line.set_data([], [])
            return condition_line, speed_line, reward_line, bars

        def update(frame):
            nonlocal state  # honsetly not sure what this does, found it on a stack overflow + it doesn't work without it so who cares what it is
            # select action + next step w/ env
            action = np.random.choice(self.action_space)
            next_state, reward, done, _ = self.step(action)

            # update the wanted performance metrics
            condition = np.mean(next_state)  
            speed = np.mean(next_state - state)  # deteroriation speed
            cumulative_reward = rewards[-1] + reward if rewards else reward

            years.append(frame)
            condition_history.append(condition)
            speed_history.append(speed)
            rewards.append(cumulative_reward)
            actions.append(self.action_map[action])  # convert action to numerical value using the action_map so that can plot the dang thing

            condition_line.set_data(years, condition_history)
            speed_line.set_data(years, speed_history)
            reward_line.set_data(years, rewards)

            ax4.clear()
            ax4.bar(years, actions, color='orange')
            ax4.set(xlabel='Year', ylabel='Actions', ylim=(0, 5))

            if done:
                ani.event_source.stop()

            state = next_state  
            return condition_line, speed_line, reward_line

        # run the animation AHHHHH !!!
        ani = FuncAnimation(fig, update, frames=episodes, init_func=init, blit=False, interval=200, repeat=False)
        plt.tight_layout()
        plt.show()
