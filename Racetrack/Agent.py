# Jane Slagle
# CS 138 - Programming Assignment 2
# Agent.py

import numpy as np
import random
from Racetrack import Racetrack

class Agent:
    '''
    Overview:
    
    Represents an agent learning how to navigate through a racetrack. The racetrack is given as input
    and the agent then creates the racetrack env that it will interact with using the Racetrack class.
    The purpose of the Agent class is to guide the agent through the racetrack env while also learning
    the best action to take based on the state, action values (the Q func values). The goal is to learn
    the optimal policy, so the policy that will tell the agent which action is best to take in each given
    state. 
    
    ***Follows off-policy MC control for estimating optimal policy algorithm given in Section 5.7
    of the book on page 111.***
    
    ----------------------------------------------------------------------------------------------------
    
    Attributes:
    
    racetrack - Inputted list of strings representing racetrack where each entry represents start line pos,
    finish line pos, valid driving pos in racetrack or boundary wall pos. The agent plugs this racetrack
    into the Racetrack class to create the racetrack env that it will then interact with. Create the 
    racetrack in the init so that the agent will have env created when instance of agent is made.
    
    x_dim, y_dim - Get dimensions of racetrack out after the racetrack has been set up. Will need for
    setting up Q, C, pi arrs.
    
    C - C(s,a), count of state-action pairs, keeps track of num times agent visited state s + taken act a.
    Same ideas as Q where have (x,y,vx,vy,ax,ay) as key and values = count of how many times specific
    state,act pair visited.
    Have C[(x,y,vx,vy,ax,ay)] = visit count
    
    Q - Q(s,a) func, expected sum of discounted rewards of taking act a in state s, following opt policy
    afterwards. Init as dict where the keys = tuples = (state, act) pairs where the state is a tuple
    rep. state of car = (x,y,vx,vy), the position followed by the velocity and the action of car = tuple
    rep act taken = (ax, ay). The values of dict = the expected sum of discounted rewards from taking
    that act in that state.
    Have Q[(x,y,vx,vy,ax,ay)] = expected sum discounted rewards
    
    pi - Policy mapping from states to actions, indiciates act agent should taken when in state s. Init 
    as dict where keys = state = tuple (x,y,vx,vy) and values = opt act take in that state = also tuple 
    rep as (ax,ay). 
    Have pi[(x,y,vx,vy)] = (ax,ay)
    
    eps - 0.1, following usual default value for it (discussed in class + book that
    this is typical val chosen for eps).
    
    discount_fac - discount factor gamma, typical value used is 0.9 (used in class + book).
    
    poss_acts - range of possible action values is -1 to 2.
    
    default_act - (0,0). Use when init dicts to make sure all dicts are populated w/ vals so that 
    don't get key errors.
    
    -----------------------------------------------------------------------------------------------------
    
    Methods:
    
    b_policy - corresponds to b, any soft policy, given in the loop for each episode in off policy
    MC control algorithm that are following from book in this Agent class. Says that b can be any soft
    policy so use eps-greedy policy since have past experience from programming assignment one with 
    that policy.
    
    generate_eps - corresponds to generate an episode using b part from the off policy MC control
    algor that are following from book in this Agent class. For each episode with MC control methods,
    want the experience from each episode where the experience from each episode = sequence of states, 
    actions, rewards from each step in the episode. Each episode lasts from the start until the car
    crosses the finish line, episode only ever ends when car crosses that finish so that is when
    terminate each episode.
    
    loop_each_eps - corresp to loop forever (for each episode) part from the off policy MC control
    algor that are following from book in this Agent class. In this loop, we generate an eps (so call
    generate_eps func), set G=0 and W=1 and then loop over each step of episode, which handle by
    calling loop_each_step_eps func.
    
    loop_each_step_eps - handles looping over each episode in reverse, upadting G, C, Q, the optimal 
    policy and using importance sampling when doing so.
    '''
    def __init__(self, racetrack):
        self.racetrack = Racetrack(racetrack)
        self.x_dim = len(racetrack[0])
        self.y_dim = len(racetrack)
        self.C = dict()
        self.Q = dict()
        self.pi = dict()
        self.eps = 0.1
        self.discount_fac = 0.9
        self.poss_acts = range(-1, 2)
        self.default_act = (0, 0) 
                            
    #dets which act agent shld take in given inputted state 
    def b_policy(self, state):
        #1st get all info out of state param. state = tuple (pos, vel) = ((x,y), (vx,vy))
        pos, vel = state
        x, y = pos         #pos is given by (x, y)
        vx, vy = vel       #vel is given by (vx, vy)
        curr_state = (x,y,vx,vy)
        
        #now do eps-greedy act selection!
        if np.random.rand() > self.eps:
            #if rand num btw 0,1 is greater than eps then exploit!
            #means choose best act to take (one that max the policy)
            #pi attribute = stores the opt act as its vals + keys in the dict = the state so to get
            #best act for the given state here, just need to pull out the value at the curr state key!
            #if haven't visited given state yet (so no best act for it yet recorded in policy dict),
            #then give (0,0) act since it means no movement
            best_act = self.pi.get((x,y,vx,vy), self.default_act)
            return best_act

        else:
            #else, explore!
            #means choose rand act out of all poss ones have
            #poss acts here = [-1, 0, 1]
            #act = rep as tuple (ax, ay)
            rand_act = random.choice([(ax, ay) for ax in self.poss_acts for ay in self.poss_acts])
            return rand_act
        
    def generate_eps(self):
        #store all of the seqs of states, acts, rewards as tuple in eps list
        eps = []   #corresp to one eps
        
        #eps only ever finishes when car crosses finish line
        while not self.racetrack.car_on_finish_line():
            state = self.racetrack.car_state()     #get state out of racetrack obj
            act = self.b_policy(state)             #now get act for state using b policy!
            rew = self.racetrack.car_take_act(act) #get reward from taking act in state
            
            #now have everything want from each step in episode so append it to list as tuple
            eps.append((state, act, rew))
            
        return eps
    
    def loop_each_eps(self, num_eps):
        #this func = specifically loops over all eps
        for eps in range(num_eps):
            #each time start a new eps, reset the racetrack env!
            self.racetrack.reset()
            
            #generate eps using b
            gen_eps = self.generate_eps()
            
            #then loop over each step of episode where update everything!!!
            self.loop_each_step_eps(gen_eps, G=0, W=1)
    
    def loop_each_step_eps(self, eps, G, W):
        #want loop over episode backwards: from time step t = T all the way down to 0
        #also want to get all info out of the episode
        for state, act, rew in reversed(eps):
            #1st update G w/ formula: G = gamma*G + R_(t+1)
            G = self.discount_fac*G + rew
            
            #now want update C + Q so need (S_t, A_t), the state act pair to be able to do so
            #(S_t, A_t) here reps the key used to access C, Q so need use the state, act that just got
            #out of the episode to create key needed for C, Q dicts 
            #key for C, Q dicts is given by (x,y,vx,vy,ax,ay)
            x, y = state[0][0], state[0][1]
            vx, vy = state[1][0], state[1][1]
            ax, ay = act
            step_key = (x,y,vx,vy,ax,ay) 
            
            #if no value set at key just found in C and Q, then init as 0 
            if step_key not in self.C:
                self.C[step_key] = 0
            if step_key not in self.Q:
                self.Q[step_key] = 0
            
            #now that have the key (S_t,A_t) can update C w/ formula: C(S_t,A_t) = C(S_t,A_t) + W
            self.C[step_key] += W
            
            #now that have key (S_t,A_t) can update Q w/ formula: 
            #Q(S_t,A_t) = Q(S_t,A_t) + W/C(S_t,A_t)*[G - Q(S_t,A_t)]
            self.Q[step_key] += (W / self.C[step_key]) * (G - self.Q[step_key])
            
            #now update policy pi using formula: pi(S_t) = argmax_a Q(S_t,a)
            #so find the max action 
            #so loop over all poss acts can take
            max_action = None              #init as None so that don't have a valid action until
                                           #actually do find one to set max_action to be
            max_q_val = -1000000000000000  #init max_q_val that can use to compare. set as really small 
                                           #init so that will update with first pass through
                
            for a_x in self.poss_acts:
                for a_y in self.poss_acts:
                    #get key val corresp to act looping through
                    key_val = (x, y, vx, vy, a_x, a_y)
                    
                    #now get Q val for that key, if no value yet then use the max one to start out w/
                    q_val = self.Q.get(key_val, max_q_val)
                    
                    #check if the Q value just found is greater than max one storing
                    if q_val > max_q_val:
                        #if it is then update the max action so know what it is
                        max_action = (a_x, a_y)
                        
                        #and also update the max_q_val
                        max_q_val = q_val
    
            #now cover if, else statements at end of algor given in book
            #if the corresp act val from policy does not equal max one just found then exit for loop
            if self.pi.get((x,y,vx,vy)) != max_action:
                #then set it to be and break
                self.pi[(x,y,vx,vy)] = max_action
                break
            else:
                #update policy value in this case too, want set it to be max in both cases!
                self.pi[(x,y,vx,vy)] = max_action
                
                #else statement says to update W w/ formula: W = W(1/b(A_t|s))
                #means need b(A_t|s) which is prob that behavior policy selects act A_t, opt act, in state
                #s. Know have 9 total actions here so w/ eps greedy b policy have: choose non best act
                #w/ prob 8/9*eps but want prob choose best one so subtract that from 1 to get what 
                #actually want
                prob_term = 1 - ((8/9) * self.eps)
                W *= 1 / prob_term
                
                
