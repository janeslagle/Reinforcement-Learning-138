# CS 138 Programming Assignment 1 - Jane Slagle

## Problem Statement
**Exercise 2.5** from **Chapter 2 of Sutton and Barto's _Reinforcement Learning_**:

> Design and conduct an experiment to demonstrate the  
> difficulties that sample-average methods have for nonstationary problems. Use a modified  
> version of the 10-armed testbed in which all the `q_*(a)` start out equal and then take  
> independent random walks (say by adding a normally distributed increment with mean 0  
> and standard deviation 0.01 to all the `q_*(a)` on each step). Prepare plots like Figure 2.2  
> for an action-value method using sample averages, incrementally computed, and another  
> action-value method using a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and  
> longer runs, say of 10,000 steps.

## Requirements
- **Python**: 3.11.5
- **NumPy**: 2.1.1
- **Matplotlib**: 3.9.2

## To Run
1. Open the `main.py` file
2. Uncomment the plotting function you want to see results from in the `if __name__ == "__main__":` block:
   - `plot_eps_greedy()` for **Figure 1** in the report
   - `plot_UCB()` for **Figure 2** in the report
3. Run `main.py` to see the plotting results

## Deliverables
A detailed description of the design process and analysis of the results can be found in _COMP-138-Programming-Assignment-1.pdf_.
