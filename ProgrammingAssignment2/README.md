# CS 138 Programming Assignment 2 - Jane Slagle

## Requirements
- **Python**: 3.11.5
- **NumPy**: 2.1.1
- **Matplotlib**: 3.9.2

## To Run
1. Open the 'main.py' file
2. Uncomment the simulation function that you want to see results from in the 'if __name__ == "__main__":' block:
  - The results are each run for 300,000 episodes.
  - 'racetrack_one_trajs = run_simulation(racetrack_one, N)' shows the trajectories on a visual representation of the same racetrack that is on the left side of Figure 5.5 in the book. The plot from running this function is given as **Figure 2** in the report.
  - 'print(racetrack_one_trajs)' prints out the dictionary of trajectories for each starting position point that is plotted in the above graph. This dictionary has keys that are the position of each starting point on the start line for the plotted racetrack and the values are the position of the car at each point in the trajectory path as well as the action taken by the car at that position. I used the results of this dictionary to analyze my results in the report.
  - 'racetrack_two_trajs = run_simulation(racetrack_two, N)' shows the trajectories on a visual representation of the same racetrack that is on the right side of Figure 5.5 in the book. The plot from running this function is given as **Figure 3** in the report.
  - 'print(racetrack_two_trajs)' is the same as 'print(racetrack_one_trajs)' but for the trajectory dictionary used to get the results for racetrack_two.
3. Run 'main.py' to see the results that you chose to run from Step (2).
