# Jane Slagle
# CS 138 - Programming Assignment 2
# main.py

import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from Racetrack import Racetrack
from Agent import Agent

def visualize_racetrack(racetrack, trajs):
    '''
    Create visual representation of inputted racetrack and plot the trajectory of the car on the
    racetrack visualization from each possible starting line position when the car is following
    the optimal policy. This plot shows the entire trajectory from each starting line position as the 
    car navigates from the start position to the finish line.
    It takes in the same racetrack representation that is inputted into the Racetrack class, so a list
    of strings that make up the racetrack layout.
    It also takes in trajs as input which is the dict of trajs returned by the car_trajectories() func.
    '''
    #make plot bigger, otherwise it's too small!
    plt.figure(figsize=(20, 15))
    
    #1st create visualization of racetrack by looping through racetrack list given, use strs to fill 
    #in plot with colors that correspond to each different part of racetrack (wall, start valid driving 
    #path, and finish line)
    for y in range(len(racetrack)):
        for x in range(len(racetrack[y])):
            if racetrack[y][x] == '#':
                #corresps to wall part of racetrack
                plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor='darkgray'))  
            elif racetrack[y][x] == 's':
                #start line part of racetrack
                plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor='lightcoral'))
            elif racetrack[y][x] == '/':
                #valid driving part of racetrack
                plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor='gainsboro'))
            elif racetrack[y][x] == '!':
                #finish line part of racetrack
                plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor='lightgreen')) 
                
    #now loop over each traj in trajs dict, plot path from each start pos
    #trajs = dict where keys = tuple = start pos of car on racetrack (x,y)
    #vals = list traj = sequence of tuples where each tuple = (pos, act taken at pos)
    for start_pos, traj in trajs.items():
        #get each position point out of traj
        #1st thing returned in traj is the position so access traj[0][x or y] to get x, y coords of pos out
        traj_pos_x = [pos_pt[0][0] for pos_pt in traj]
        traj_pos_y = [pos_pt[0][1] for pos_pt in traj]
        
        #bc have racetrack flipped so (0,0) is bottom left, need flip the y coords of pos so that they
        #have correct orientation on plot
        traj_pos_y = [len(racetrack) - 1 - y for y in traj_pos_y]
        
        #now that have the x, y position pts for each traj point, plot each point on graph
        #mark each point w/ o so that show up as circle on graph and can see each position point car
        #took on each traj path
        #connect all of the pts in traj path w/ a line
        plt.plot(traj_pos_x, traj_pos_y, marker = 'o', label=f'{start_pos} trajectory', linewidth = 2)
    
    plt.title('Trajectories From Each Starting State Following Optimal Policy') 
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()  #invert y axis to follow same pattern that (0,0) is in bottom left here
    plt.legend()
    
    #dont want nums to show up on x, y axis bc those nums have no meaning with the racetrack
    plt.xticks([])
    plt.yticks([])  
    
    plt.show()
    
def run_simulation(racetrack, N):
    #create agent w/ inputted racetrack
    agent = Agent(racetrack)
    
    #run simulation of car through racetrack with inputted N where N = num episodes to run simulation w/
    agent.loop_each_eps(N)
    
    #get trajectories for each starting line pos from racetrack simulation, pass in the learned policy 
    #from Agent class
    trajs = agent.racetrack.car_trajectories(agent.pi)
    
    #plot the trajectory results now!
    visualize_racetrack(racetrack, trajs)
    
    #return the trajs so that can have the position and actions taken at each position to use to analyze
    #results in report
    return trajs
    
if __name__ == '__main__':
    racetrack_one = [
        '##################',
        '####/////////////!',        
        '###//////////////!',
        '###//////////////!',
        '##///////////////!',
        '#////////////////!',
        '#////////////////!',
        '#//////////#######',
        '#/////////########',
        '#/////////########',
        '#/////////########',
        '#/////////########',
        '#/////////########',
        '#/////////########',
        '#/////////########',
        '##////////########',
        '##////////########',
        '##////////########',
        '##////////########',
        '##////////########',
        '##////////########',
        '##////////########',
        '##////////########',
        '###///////########',
        '###///////########',
        '###///////########',
        '###///////########',
        '###///////########',
        '###///////########',
        '###///////########',
        '####//////########',
        '####//////########',
        '####ssssss########'
    ]
    
    racetrack_two = [
        '#################################',
        '#################///////////////!',
        '##############//////////////////!',
        '#############///////////////////!',
        '############////////////////////!',
        '############////////////////////!',
        '############////////////////////!',
        '############////////////////////!',
        '#############///////////////////!',
        '##############//////////////////!',
        '###############////////////////##',
        '###############/////////////#####',
        '###############////////////######',
        '###############//////////########',
        '###############/////////#########',
        '##############//////////#########',
        '#############///////////#########',
        '############////////////#########',
        '###########/////////////#########',
        '##########//////////////#########',
        '#########///////////////#########',
        '########////////////////#########',
        '#######/////////////////#########',
        '######//////////////////#########',
        '#####///////////////////#########',
        '####////////////////////#########',
        '###/////////////////////#########',
        '##//////////////////////#########',
        '#///////////////////////#########',
        '#///////////////////////#########',
        '#sssssssssssssssssssssss#########'
    ]
    
    N = 300000
    #racetrack_one_trajs = run_simulation(racetrack_one, N)
    #print(racetrack_one_trajs)
    
    racetrack_two_trajs = run_simulation(racetrack_two, N)
    print(racetrack_two_trajs)
   
    pass

