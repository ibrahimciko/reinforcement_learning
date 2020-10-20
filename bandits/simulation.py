#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:33:42 2020

@author: ibrahim
"""

import numpy as np

class Simulation():
    
    def __init__(self,time_horizon,num_simulation,num_bandits,num_arms):
        self.T = time_horizon
        self.num_bandits = num_bandits
        self.K = num_arms
        self.num_simulation = num_simulation
        self.avg_rewards = [np.zeros(time_horizon) for i in range(num_bandits)]
        self.avg_regrets = [np.zeros(time_horizon) for i in range(num_bandits)]
        self.avg_num_chosen_arms = [np.zeros(self.K) for i in range(num_bandits)]
        
        
    def update_avg_reward(self,current_rewards,n):
        for i in range(self.num_bandits):
            self.avg_rewards[i] += ((np.array(current_rewards[i]) - self.avg_rewards[i])/n  ).flatten()
        
        
    def update_avg_regret(self,current_regrets,n):
        for i in range(self.num_bandits):
            self.avg_regrets[i] += ((np.array(current_regrets[i]) - self.avg_regrets[i])/n).flatten()
        
    def update_avg_num_chosen_arms(self,current_chosen_arms,n):
        for i in range(self.num_bandits):
            self.avg_num_chosen_arms[i] += ((np.array(current_chosen_arms[i]) - self.avg_num_chosen_arms[i])/n).flatten()
        
        
    def start_simulation(self,environment):
        environment.clear_interaction()
        for i in range(self.num_simulation):
            for j in range(self.T):
                environment.interaction()   
            #print("One run completed!")
            #print(environment.bandits[0].perceived_probs)
            self.update_avg_reward(current_rewards=environment.reward_history, n=i+1)
            self.update_avg_regret(current_regrets=environment.regrets, n=i+1) 
            self.update_avg_num_chosen_arms(current_chosen_arms=environment.num_chosen_arms, n=i+1)
            environment.clear_interaction()
        
            if (i + 1) %1000 == 0:
                print(f'Simulation Number: {i+1}')
        
        #self.avg_regrets = [i/self.T for i in self.avg_regrets]
        
        
        
# Debug
if __name__ == "__main__":
    from environment import Environment
    from greedy import GreedyBandit
    from arm import Arm
    from thompson import ThompsonBandit
    
    K = 4
    b1 = ThompsonBandit("thompson", default_priors = [[1,1] for i in range(K)])
    b2 = GreedyBandit("greedy",default_priors =[0.5 for i in range(K)])
    a1 = Arm("a1",is_life_style=True, true_life_style_prob=[0.8,0.2,0.2,0.2]) 
    a2 = Arm("a2",is_life_style=True, true_life_style_prob=[0.2,0.8,0.2,0.2])
    a3 = Arm("a3",is_life_style=True, true_life_style_prob=[0.2,0.2,0.8,0.2]) 
    a4 = Arm("a4",is_life_style=True, true_life_style_prob=[0.2,0.2,0.2,0.8]) 
    #a3 = Arm("a3",0.6)
    #a4 = Arm("a4",0.65)
    pop1 = [0.25,0.25,0.25,0.25] #a
    pop2 = [0.99,0.01,0.0,0.0]   #b
    pop3 = [0.33,0.33,0.33,0.01] #c
    pop4 = [0.60,0.25,0.15,0.0]  #d
    env = Environment([b1,b2],[a1,a2,a3,a4],True,pop4)
    
    #Check whether perceived probabilities change!
    #start
    sim = Simulation(1000,10000,len(env.bandits),K)
    sim.start_simulation(env)



    
