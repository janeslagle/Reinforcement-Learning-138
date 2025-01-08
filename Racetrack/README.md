# CS 138 Programming Assignment 2 - Jane Slagle

## Problem Statement
**Exercise 5.12** from **Chapter 5 of Sutton and Barto's _Reinforcement Learning_**:

> Consider driving a race car around a turn  
> like those shown in Figure 5.5. You want to go as fast as possible, but not so fast as  
> to run off the track. In our simplified racetrack, the car is at one of a discrete set of  
> grid positions, the cells in the diagram. The velocity is also discrete, a number of grid  
> cells moved horizontally and vertically per time step. The actions are increments to the  
> velocity components. Each may be changed by +1, -1, or 0 in each step, for a total of  
> nine (3×3) actions. Both velocity components are restricted to be nonnegative and less  
> than 5, and they cannot both be zero except at the starting line. Each episode begins  
> in one of the randomly selected start states with both velocity components zero and  
> ends when the car crosses the finish line. The rewards are -1 for each step until the car  
> crosses the finish line. If the car hits the track boundary, it is moved back to a random  
> position on the starting line, both velocity components are reduced to zero, and the  
> episode continues. Before updating the car’s location at each time step, check to see if  
> the projected path of the car intersects the track boundary. If it intersects the finish line,  
> the episode ends; if it intersects anywhere else, the car is considered to have hit the track  
> boundary and is sent back to the starting line. To make the task more challenging, with  
> probability 0.1 at each time step the velocity increments are both zero, independently of  
> the intended increments. Apply a Monte Carlo control method to this task to compute  
> the optimal policy from each starting state. Exhibit several trajectories following the  
> optimal policy (but turn the noise off for these trajectories).

## Requirements
- **Python**: 3.11.5
- **NumPy**: 2.1.1
- **Matplotlib**: 3.9.2

## To Run
1. Open the 'main.py' file
2. Uncomment the simulation function that you want to see results from in the 'if __name__ == "__main__":' block:
  - The results are each run for 300,000 episodes.
  - 'racetrack_one_trajs = run_simulation(racetrack_one, N)' shows the trajectories on a visual representation of the same racetrack that is on the left side of Figure 5.5 in the book. The plot from running this function is given as **Figure 3** in the report.
  - 'print(racetrack_one_trajs)' prints out the dictionary of trajectories for each starting position point that is plotted in the above graph. This dictionary has keys that are the position of each starting point on the start line for the plotted racetrack and the values are the position of the car at each point in the trajectory path as well as the action taken by the car at that position. I used the results of this dictionary to analyze my results in the report, specifically in **Table 1**.
  - 'racetrack_two_trajs = run_simulation(racetrack_two, N)' shows the trajectories on a visual representation of the same racetrack that is on the right side of Figure 5.5 in the book. The plot from running this function is given as **Figure 4** in the report.
  - 'print(racetrack_two_trajs)' is the same as 'print(racetrack_one_trajs)' but for the trajectory dictionary used to get the results for racetrack_two.
3. Run 'main.py' to see the results that you chose to run from Step (2).

## Deliverables
A detailed description of the design process and analysis of the results can be found in _COMP-138-Programming-Assignment-2.pdf_.
