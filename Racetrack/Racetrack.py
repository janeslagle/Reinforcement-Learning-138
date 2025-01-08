# Jane Slagle
# CS 138 - Programming Assignment 2
# Racetrack.py

import numpy as np
import random

class Racetrack:
    '''
    Overview:
    
    Builds the racetrack environment by defining the layout of the racetrack, controlling the movement of
    a car on the racetrack based on actions that the car is able to take, detects possible collisions of
    the car on the racetrack with walls, keeps track of whether the car crosses the finish line and also
    computes the trajectory of the car from each starting position on the racetrack based on an optimal
    policy found in the Agent class.
    
    ----------------------------------------------------------------------------------------------------
    
    Attributes:
    
    racetrack - Inputted list of strings representing racetrack where each entry represents start line pos,
    finish line pos, valid driving pos in racetrack or boundary wall pos.
    
    start_line - Stores valid positions from racetrack working w/ that make up the start line on the track.
    Want bc start at random position on start line each time.
    
    finish_line - Stores valid positions from racetrack working w/ that make up the finish line on the
    track. Want bc then can easily check if car has crossed finish line.
    
    min_vel - Told in problem that velocity of car cannot be below 0. 
    
    max_vel - Told in problem that velocity of car cannot be above 4.
    
    num_rows, num_cols - Set and store size of racetrack after it has been built in build_racetrack method.
    
    -----------------------------------------------------------------------------------------------------
    
    Methods:
    
    build_racetrack - Takes the inputted list of strings, racetrack, uses it to build a 2D np array 
    numerical representation of the racetrack. Call in init so that valid racetrack np array is created
    when create instance of Racetrack class.
    
    reset - If car hits track boundary (wall) during episode, want reset car's position to random pos
    on start line, reset velocity (horiz + vert) comps to both be 0. Call in init so that each time 
    create instance of Racetrack, start at random position and with 0 velocity like want to.
    
    update_vel - Updates car's velocity based on inputted action where the inputted action is a tuple (x,y)
    representing how much the velocity changes when that action is taken.
    
    will_car_hit_wall - Bool that tells us if car hit wall at inputted given position.
    
    will_car_cross_finish_line - Bool that tells us if car crossed finish line at inputted given position.
    
    update_pos - Update car's position based on it's current velocity and returns corresponding reward
    based on the position so that will know the wanted reward when the car actually takes action in 
    car_take_act func().

    car_take_act - Executes an action for car on racetrack, updates the state of the car by updating both
    it's position and velocity. The input is the action taken by the car.
    
    car_on_finish_line - Bool that tells us if the current position of car is found in the finish_line 
    list. If yes, then we know the car has crossed the finish line. If not, then the car has not. 
    Used in car_trajectories func bc easier to check than using will_car_cross_finish_line bc have to
    pass in the position with that function.
    
    car_state - Get current state of car out. State of car is given by (car position, car velocity) where
    both position and velocity are 2D.
    
    car_trajectories - Compute the car's trajectories for each starting line position using an optimal
    policy learned from the Agent class. These trajectories are what we ultimately want to return and 
    plot as results
    '''
    def __init__(self, racetrack):
        self.racetrack = None    #init as None bc will set up in build_racetrack func
        self.start_line = []
        self.finish_line = []
        self.min_vel = 0
        self.max_vel = 4
        self.build_racetrack(racetrack) 
        self.num_rows = self.racetrack.shape[1]   
        self.num_cols = self.racetrack.shape[0]
        self.reset()

    def build_racetrack(self, racetrack):
        #get dims of racetrack out
        num_rows = len(racetrack)
        num_cols = len(racetrack[0])
    
        #init racetrack as all 0s, then fill in based on list strs given
        self.racetrack = np.zeros((num_cols, num_rows))
    
        #populate racetrack arr w/ nums that will rep each poss diff position can have on racetrack
        for y in range(num_rows):
            for x in range(num_cols):
                racetrack_cell = racetrack[y][x]    #current instance on racetrack grid
    
                #now check what position current racetrack_cell has in racetrack
                if racetrack_cell == '#':
                    self.racetrack[x, y] = 0        #0 reps invalid part of racetrack: boundary wall
                elif racetrack_cell == 's':
                    self.racetrack[x, y] = 1        #1 reps valid start line
                    self.start_line.append((x, y))  #populate list of valid start line positions
                elif racetrack_cell == '/':
                    self.racetrack[x, y] = 2        #2 reps valid part of racetrack that can drive on
                elif racetrack_cell == '!':
                    self.racetrack[x, y] = 3        #3 reps valid finish line
                    self.finish_line.append((x, y)) #populate list of valid finish line positions
                
        #need, otherwise racetrack is upside down when visualize it
        self.racetrack = np.fliplr(self.racetrack)

        #need, otherwise y values are flipped from what they should be on graph when visualize it
        self.start_line = [(x, num_rows - 1 - y) for (x, y) in self.start_line]
        self.finish_line = [(x, num_rows - 1 - y) for (x, y) in self.finish_line]
        
    def reset(self):
        self.pos = np.array(random.choice(self.start_line))
        self.vel = np.array([0, 0])
        
    def update_vel(self, act):
        #each act = increments vel components so update curr vel by adding on act taken
        self.vel += np.array(act)
        
        #need make sure vel stays btw 0 + 4 so use np.clip for that
        #also before actually set vel to be the updated one, need make sure that the new vel is valid
        #so store it as projected_vel before actually set vel to be it
        proj_vel = np.clip(self.vel, self.min_vel, self.max_vel)

        #need make sure vel = only (0,0) at start line pos so make sure it's not (0,0) at non-start line
        #position before set vel equal to projected one
        if np.array_equal(proj_vel, [0,0]):
            #check if car on start line, if has vel (0,0) not on start line then reset car's pos bc not 
            #valid
            if not any(np.array_equal(self.pos, start) for start in self.start_line):
                self.reset()
        else:
            #otherwise, vel is actually valid so can set the vel to be that!!!
            self.vel = proj_vel
            
    def will_car_hit_wall(self, pos):
        #have defined racetrack in build_racetrack func so that car is on wall if value of 
        #racetrack at that pos is equal to 0
        x = pos[0]
        y = pos[1]
        
        return self.racetrack[x, y] == 0
    
    def will_car_cross_finish_line(self, pos):
        #have defined racetrack in build_racetrack func so that car is on finish line if value of 
        #racetrack at that pos is equal to 3
        x = pos[0]
        y = pos[1]
        
        return self.racetrack[x, y] == 3
            
    def update_pos(self):   
        #before actually update position, calc projected pos are finding + check if going to 
        #intersect wall or finish line
        proj_pos = self.pos + self.vel
        
        #check if proj pos is out of valid bounds of track (had some index errs so accounting for that
        #with this)
        #if car out of bounds then reset it bc not valid pos + return -1 to indicate so bc want return
        #-1 each time car doesn't cross finish line
        if not (0 <= proj_pos[0] < self.num_cols and 0 <= proj_pos[1] < self.num_rows):
            self.reset()
            return -1
        
        #check if proj pos going to hit wall. if does, reset pos + vel + return -1
        if self.will_car_hit_wall(proj_pos):
            self.reset()
            return -1
        
        #check if proj pos going to cross finish line. if does, update the pos to reflect that car now on
        #finish line + return 0
        if self.will_car_cross_finish_line(proj_pos):
            self.pos = proj_pos
            return 0

        #update pos for case when don't hit wall or cross finish line + return -1 bc didn't hit finish line
        self.pos = proj_pos
        return -1
    
    def car_take_act(self, act):
        #1st update vel bc update car's pos based on vel so need vel before can get the pos
        self.update_vel(act)
        
        #then update pos of car based on the new vel
        new_car_pos = self.update_pos()
        
        #then return wanted reward depending on what the new car pos returned is (handle all of this
        #internally in the update_pos() func, it either returns 0 or -1 based on what the new pos is)
        #either returns -1 or 0 w/ 0 indiciating that car crossed finish line and -1 indicating car hit
        #wall or still driving on track
        return new_car_pos
    
    def car_on_finish_line(self):
        #check if the current position of car is found in finish_line list or not
        return tuple(self.pos) in self.finish_line
    
    #state of car = (pos, vel) where pos = (x,y) and vel has horiz + vert comps (vx, vy)
    #so returns both (x,y) and (vx, vy)
    def car_state(self):
        #if dont't have as copy then get stuck in inf loop so return as copies to avoid modifying both
        #when don't want to be
        return self.pos.copy(), self.vel.copy()

    #returns dict where keys are each starting pos + vals = trajectory from start to finish line for each
    #key (starting pos) given as list
    #so each trajectory = list of tuples = (pos, act) so that know where the car is at each point and
    #so that also know what action car took from each point (so that are able toget some insight into 
    #if slowed down, sped up, stayed same speed, etc. when interpret results)
    def car_trajectories(self, opt_policy):
        trajs = dict()

        #loop through all of starting line positions!!!
        for start_pos in self.start_line:
            #don't want start at random point now, want start specifically at this starting line spot
            #so set pos of racetrack to be that position, make sure velocity is (0,0) to indicate 
            #that starting
            self.pos = np.array(start_pos)
            self.vel = np.array([0,0])
            
            #store sequence of pos,act pairs that will form traj for curr starting pos's path
            traj = []
            
            #get path of car from that starting pos until it reaches finish line (bc then it's path
            #is over)
            while not self.car_on_finish_line():
                #get state (pos + vel) of car at each point in it's path
                car_state = self.car_state()
                
               #pull pos, vel out of state
                x, y = car_state[0][0], car_state[0][1]
                vx, vy = car_state[1][0], car_state[1][1]
                
                #now that have pos + vel of car at current point on its path, get opt action that car
                #shld take using the inputted opt policy
                #key to get opt act out of policy is (x,y,vx,vy)
                #give default act of (0,0) if no val for that key yet to avoid key errs!
                opt_act = opt_policy.get((x,y,vx,vy), (0,0))
                
                #add pos, act to traj list to keep track of what car doing on its path
                traj.append((tuple(self.pos), opt_act))
                
                #then actually have car take act! 
                self.car_take_act(opt_act)
            
            #know that car has crossed finish line now so need add final point to its traj path!
            #when car crossed finish line, car is done moving so takes no action so put None to reflect
            #that
            traj.append((tuple(self.pos), None))
            
            #add traj list to dict for each starting pos to get complete dict want!!!
            trajs[tuple(start_pos)] = traj
        
        return trajs
 
 
