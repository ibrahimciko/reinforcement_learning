#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:30:21 2020

@author: ibrahim
"""

import random
from copy import deepcopy

class GreedyBandit():
    is_greedy = True
    
    def __init__(self,name,default_priors):
        
        """
    
        Parameters
        ----------
        name: String
        
        default_priors: list - of probabilities
        
        Attributes
        ----------
        
        default_priors: list -of probabilities: priors to use when reset the bandit
        
        perceived_priors: list-of probabilities: to be used when updating parameters
        
        K: Number of arms, inferred from length of default_priors
        
        choice_history: list - of indices of chosen arms
        
        num_chosen_arms: list - of numbers where each number denote how many times
                          the arm with the corresponding index is  played
                          
        reward_history: list - of 1's and 0's where 1 means the reward attained
        
        regret: list - Takes 1 if the optimal arm is not chosen. It can computed only when 
                we know true reward probabilities of the arms
        
        data: dict  of the attributes, initalized to empty dict
        

        """
        
        self.name = name
        self.default_priors = default_priors
        self.perceived_probs = deepcopy(self.default_priors)
        self.K = len(default_priors)
        self.choice_history = []
        self.num_chosen_arms = [1 for i in range(self.K)]
        self.reward_history = []
        self.regret = []
        self.data = {}
        
    def choose_arm(self):
        """
        Method to choose an arm based on perceived probabilities.

        Returns
        -------
        choice : Integer - Index of the chosen arm
        
        """
        
        m = max(self.perceived_probs)
        maximum_indices = [i for i,j in enumerate(self.perceived_probs) if j == m]
        
        if len(maximum_indices) > 1:
            choice = random.choice(maximum_indices)
        else:
            choice = maximum_indices[0]  
            
        self.update_choice_history(choice)
        return choice
    
    def update_choice_history(self,choice):
        """
        Appends the chosen index to the self.choice_history
        
        Parameters
        ----------
        choice : Integer - Index of the chosen arm
        """
        
        self.num_chosen_arms[choice]+= 1
        self.choice_history.append(choice)
            
    def observe_outcome(self,ctr,choice):
        """
        
        Parameters
        ----------
        ctr : Integer - ClickThrough Takes 0 or 1 same as reward
        
        choice : Integer - Index of the chosen arm

        """
        self.update_reward_history(ctr)
        self.update_perceived_probs(ctr,choice) 

    def update_reward_history(self,ctr):
        """
        Appends the ctr to self.reward_history

        Parameters
        ----------
        ctr : Integer - Takes 0 or 1

        """
        
        self.reward_history.append(ctr)
    
    def update_perceived_probs(self,ctr,choice):
        """
        Update the perceived probability of the chosen arm

        Parameters
        ----------
        ctr : Integer - 1 or 0
        
        choice : Integer - index of the chosen arm

        """
        self.perceived_probs[choice] +=  (ctr - self.perceived_probs[choice])/self.num_chosen_arms[choice]
    
    def append_data_in_dict(self):
        """
        Put the attributes of the Bandit object into a dictionary format

        """
        self.data = {
            "perceived_probs": self.perceived_probs,
            "choice_history": self.choice_history,
            "num_chosen_arms": self.num_chosen_arms,
            "reward_history": self.reward_history,
            "regret": self.regret
            }
    
    def reset_priors(self):
        """
        Reset the priors to the default priors: Useful for simulation
        
        """
        
        self.perceived_probs = deepcopy(self.default_priors)
        
        
    def delete_history(self):
        """
        Method to clear bandit data, and start from the initial belief - intial_probs
        It clears the following attributes:
            
                a-) reward_history
                b-) num_chosen_arms
                c-) choice_gistory
                d-) perceived_probs
                e-) regret
                f-) data

        """
        self.reset_priors()
        self.reward_history.clear()
        self.choice_history.clear()
        self.regret.clear()
        self.data.clear()
        
        for i in range(self.K):
            self.num_chosen_arms[i] = 1
    
    def __repr__(self):
        return self.name
  

    # Debugging
if __name__ == "__main__":
    default_probs = [0.5,0.5]
    g1 = GreedyBandit("b1", default_probs)        
    g2 = GreedyBandit("b2", default_probs)
    g1.name
    g1.choice_history
    g1.reward_history
    g1.regret  
    g1.delete_history()
    g1.data
    g1.name
    g1.choice_history    
    
    #check methods
    choice = g1.choose_arm()
    print(choice)
    g1.choice_history
    g1.num_chosen_arms
    g1.observe_outcome(1, choice)
    g1.perceived_probs
    g1.reward_history
